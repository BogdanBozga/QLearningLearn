import random

class GridWorld:
    def __init__(self):

        self.positionX = 0
        self.positionY = 2
        self.done = False
        self.updateState()
        self.nr_steps = 0

    def updateState(self):
        self.state = self.positionX + self.positionY * 3


    def info(self):
        """
        Print info
        """
        print(
            """
        0 - Up
        1 - Right
        2 - Down
        3 - Left

        +---+---+---+
        |   |   | G |
        +---+---+---+
        |   |   |   |
        +---+---+---+
        | S |   |   |
        +---+---+---+
        G at state 2, S at state 6
        X+n*Y

        """)

    def step(self, action):
        self.nr_steps += 1
        if self.positionX == 2 and self.positionY == 0:
            self.done = True
            return self.state, 5, self.done, self.nr_steps

        match action:
            case 0:
                if self.positionY > 0:
                    self.positionY -= 1
            case 1:
                if self.positionX < 2:
                    self.positionX += 1
            case 2:
                if self.positionY < 2:
                    self.positionY += 1
            case 3:
                if self.positionX > 0:
                    self.positionX -= 1
        self.updateState()

        return self.state, random.randint(-12, 10), self.done, self.nr_steps


    def reset(self):
        self.positionX = 0
        self.positionY = 2
        self.updateState()
        self.done = False
        self.nr_steps = 0
        return self.state