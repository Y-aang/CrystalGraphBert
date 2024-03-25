from torch.utils.tensorboard import SummaryWriter
import os
import torch

class TensorBoardWriter:
    def __init__(self, log_dir, accumulation_steps=10):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.accumulated_data = {}

    def write_dict(self, data_dict, step):
        for key, value in data_dict.items():
            if key not in self.accumulated_data:
                self.accumulated_data[key] = []
            self.accumulated_data[key].append(value)

        self.current_step += 1
        if self.current_step % self.accumulation_steps == 0:
            averaged_data = {key: sum(values) / len(values) for key, values in self.accumulated_data.items()}
            self._write_data(averaged_data, step)
            self.accumulated_data = {}
            self._print_log(averaged_data, step)

    def _write_data(self, data_dict, step):
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                self.writer.add_scalar(key, value.item(), step)
            elif isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)
            else:
                raise ValueError(f"Unsupported data type for key '{key}': {type(value)}")

    def _print_log(self, data_dict, step):
        log_str = f"Step {step} | "
        for key, value in data_dict.items():
            log_str += f"{key}: {value} | "
        print(log_str)

    def close(self):
        self.writer.close()