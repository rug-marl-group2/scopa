from tensorboardX import SummaryWriter


class TLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        # Robustly capture log directory for different tensorboardX versions
        self._log_dir = getattr(self.writer, 'logdir', getattr(self.writer, 'log_dir', log_dir))
        self.simulation_clock = 0
        self.scopas_log = [0, 0, 0 , 0]
        self.counter = 0
        print(f"Logging to {self._log_dir}")

    def flush(self):
        self.writer.flush()

    def scopa(self, player):
        i = int(player.name[-1])
        self.scopas_log[i] += 1
        self.writer.add_scalars("Scopas pp", {'player_0': self.scopas_log[0], 
                                              'player_1': self.scopas_log[1],
                                              'player_2': self.scopas_log[2],
                                              'player_3': self.scopas_log[3]}, self.simulation_clock)
    

    def get_log_dir(self):
        return self._log_dir
    
    def add_tick(self):
        self.simulation_clock += 1

    def record_step(self, n_players = 4):
        self.counter += 1

        if self.counter == n_players:
            self.add_tick()
            self.counter = 0

    def close(self):
        self.writer.close()
        print("Logging closed")
