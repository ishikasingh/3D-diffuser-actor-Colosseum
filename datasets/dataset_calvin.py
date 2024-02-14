from collections import defaultdict, Counter
import itertools
import math
import random
from pathlib import Path

import torch

from .dataset_engine import RLBenchDataset
from .utils import Resize, TrajectoryInterpolator
from utils.utils_with_calvin import to_relative_action, convert_rotation


class CalvinDataset(RLBenchDataset):

    def __init__(
        self,
        # required
        root,
        instructions=None,
        # dataset specification
        taskvar=[('close_door', 0)],
        max_episode_length=5,
        cache_size=0,
        max_episodes_per_task=100,
        num_iters=None,
        cameras=("wrist", "left_shoulder", "right_shoulder"),
        # for augmentations
        training=True,
        image_rescale=(1.0, 1.0),
        # for trajectories
        return_low_lvl_trajectory=False,
        dense_interpolation=False,
        interpolation_length=100,
        relative_action=True
    ):
        self._cache = {}
        self._cache_size = cache_size
        self._cameras = cameras
        self._max_episode_length = max_episode_length
        self._num_iters = num_iters
        self._training = training
        self._taskvar = taskvar
        self._return_low_lvl_trajectory = return_low_lvl_trajectory
        if isinstance(root, (Path, str)):
            root = [Path(root)]
        self._root = [Path(r).expanduser() for r in root]
        self._relative_action = relative_action

        # For trajectory optimization, initialize interpolation tools
        if return_low_lvl_trajectory:
            assert dense_interpolation
            self._interpolate_traj = TrajectoryInterpolator(
                use=dense_interpolation,
                interpolation_length=interpolation_length
            )

        # Keep variations and useful instructions
        self._instructions = instructions
        self._num_vars = Counter()  # variations of the same task
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if data_dir.is_dir():
                self._num_vars[task] += 1

        # If training, initialize augmentation classes
        if self._training:
            self._resize = Resize(scales=image_rescale)

        # File-names of episodes per-task and variation
        episodes_by_task = defaultdict(list)
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if not data_dir.is_dir():
                print(f"Can't find dataset folder {data_dir}")
                continue
            npy_episodes = [(task, var, ep) for ep in data_dir.glob("*.npy")]
            dat_episodes = [(task, var, ep) for ep in data_dir.glob("*.dat")]
            pkl_episodes = [(task, var, ep) for ep in data_dir.glob("*.pkl")]
            episodes = npy_episodes + dat_episodes + pkl_episodes
            # Split episodes equally into task variations
            if max_episodes_per_task > -1:
                episodes = episodes[
                    :max_episodes_per_task // self._num_vars[task] + 1
                ]
            if len(episodes) == 0:
                print(f"Can't find episodes at folder {data_dir}")
                continue
            episodes_by_task[task] += episodes

        # Collect and trim all episodes in the dataset
        self._episodes = []
        self._num_episodes = 0
        for task, eps in episodes_by_task.items():
            if len(eps) > max_episodes_per_task and max_episodes_per_task > -1:
                eps = random.sample(eps, max_episodes_per_task)
            self._episodes += eps
            self._num_episodes += len(eps)

        print(f"Created dataset from {root} with {self._num_episodes}")

    def __getitem__(self, episode_id):
        """
        the episode item: [
            [frame_ids],  # we use chunk and max_episode_length to index it
            [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256)
                obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
            [action_tensors],  # wrt frame_ids, (1, 8)
            [camera_dicts],
            [gripper_tensors],  # wrt frame_ids, (1, 8)
            [trajectories]  # wrt frame_ids, (N_i, 8)
        ]
        """
        episode_id %= self._num_episodes
        task, variation, file = self._episodes[episode_id]

        # Load episode
        episode = self.read_from_cache(file)
        if episode is None:
            return None

        # Dynamic chunking so as not to overload GPU memory
        chunk = random.randint(
            0, math.ceil(len(episode[0]) / self._max_episode_length) - 1
        )

        # Get frame ids for this chunk
        frame_ids = episode[0][
            chunk * self._max_episode_length:
            (chunk + 1) * self._max_episode_length
        ]

        # Get the image tensors for the frame ids we got
        states = torch.stack([
            episode[1][i] if isinstance(episode[1][i], torch.Tensor)
            else torch.from_numpy(episode[1][i])
            for i in frame_ids
        ])

        # Camera ids
        if episode[3]:
            cameras = list(episode[3][0].keys())
            assert all(c in cameras for c in self._cameras)
            index = torch.tensor([cameras.index(c) for c in self._cameras])
            # Re-map states based on camera ids
            states = states[:, index]

        # Split RGB and XYZ
        rgbs = states[:, :, 0, :, 20:180, 20:180]
        pcds = states[:, :, 1, :, 20:180, 20:180]
        rgbs = self._unnormalize_rgb(rgbs)

        # Get action tensors for respective frame ids
        action = torch.cat([episode[2][i] for i in frame_ids])

        # Sample one instruction feature
        if self._instructions is not None:
            instr_ind = episode[6][0]
            instr = torch.as_tensor(self._instructions[instr_ind])
            instr = instr.repeat(len(rgbs), 1, 1)
        else:
            instr = torch.zeros((rgbs.shape[0], 53, 512))

        # Get gripper tensors for respective frame ids
        gripper = torch.cat([episode[4][i] for i in frame_ids])

        # gripper history
        if len(episode) > 7:
            gripper_history = torch.cat([
                episode[7][i] for i in frame_ids
            ], dim=0)
        else:
            gripper_history = torch.stack([
                torch.cat([episode[4][max(0, i-2)] for i in frame_ids]),
                torch.cat([episode[4][max(0, i-1)] for i in frame_ids]),
                gripper
            ], dim=1)

        # Low-level trajectory
        traj, traj_lens = None, 0
        if self._return_low_lvl_trajectory:
            if len(episode) > 5:
                traj_items = [
                    self._interpolate_traj(episode[5][i]) for i in frame_ids
                ]
            else:
                traj_items = [
                    self._interpolate_traj(
                        torch.cat([episode[4][i], episode[2][i]], dim=0)
                    ) for i in frame_ids
                ]
            max_l = max(len(item) for item in traj_items)
            traj = torch.zeros(len(traj_items), max_l, traj_items[0].shape[-1])
            traj_lens = torch.as_tensor(
                [len(item) for item in traj_items]
            )
            for i, item in enumerate(traj_items):
                traj[i, :len(item)] = item
            traj_mask = torch.zeros(traj.shape[:-1])
            for i, len_ in enumerate(traj_lens.long()):
                traj_mask[i, len_:] = 1

        # Augmentations
        if self._training:
            if traj is not None:
                for t, tlen in enumerate(traj_lens):
                    traj[t, tlen:] = 0
            modals = self._resize(rgbs=rgbs, pcds=pcds)
            rgbs = modals["rgbs"]
            pcds = modals["pcds"]

        # Compute relative action
        if self._relative_action and traj is not None:
            rel_traj = torch.zeros_like(traj)
            for i in range(traj.shape[0]):
                for j in range(traj.shape[1]):
                    rel_traj[i, j] = torch.as_tensor(to_relative_action(
                        traj[i, j].numpy(), traj[i, 0].numpy(), clip=False
                    ))
            traj = rel_traj

        # Convert Euler angles to Quarternion
        action = torch.cat([
            action[..., :3],
            torch.as_tensor(convert_rotation(action[..., 3:6])),
            action[..., 6:]
        ], dim=-1)
        gripper = torch.cat([
            gripper[..., :3],
            torch.as_tensor(convert_rotation(gripper[..., 3:6])),
            gripper[..., 6:]
        ], dim=-1)
        gripper_history = torch.cat([
            gripper_history[..., :3],
            torch.as_tensor(convert_rotation(gripper_history[..., 3:6])),
            gripper_history[..., 6:]
        ], dim=-1)
        if traj is not None:
            traj = torch.cat([
                traj[..., :3],
                torch.as_tensor(convert_rotation(traj[..., 3:6])),
                traj[..., 6:]
            ], dim=-1)

        ret_dict = {
            "task": [task for _ in frame_ids],
            "rgbs": rgbs,  # e.g. tensor (n_frames, n_cam, 3+1, H, W)
            "pcds": pcds,  # e.g. tensor (n_frames, n_cam, 3, H, W)
            "action": action,  # e.g. tensor (n_frames, 8), target pose
            "instr": instr,  # a (n_frames, 53, 512) tensor
            "curr_gripper": gripper,
            "curr_gripper_history": gripper_history
        }
        if self._return_low_lvl_trajectory:
            ret_dict.update({
                "trajectory": traj,  # e.g. tensor (n_frames, T, 8)
                "trajectory_mask": traj_mask.bool()  # tensor (n_frames, T)
            })
        return ret_dict
