def setup(self):
    self.model = QWalkerModel()


def act(self, game_state: dict):
    return self.model.propose_action(game_state)