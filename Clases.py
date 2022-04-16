## Imports
import numpy as np
import pygame

## Clases
class pre_wind:
    def __init__(self):
        self.data = []
        self.count = 0
        self.e1 = 'Nada'
        self.e2 = 'Nada'

    def actualizar(self, Var = 'Nada'):
        self.count += 1
        if Var != 'Nada':
            if self.e1 == 'Nada':
                self.e1 = Var
                self.count = 0
            else:
                self.e2 = Var

        if self.e2 != 'Nada':

            if self.e1 == 'Parpadeo' and self.e2 == 'Derecha':
                pygame.mixer.init()
                pygame.mixer.music.load("p1.mp3")
                pygame.mixer.music.play()

            elif self.e1 == 'Izquierda' and self.e2 == 'Izquierda':
                pygame.mixer.init()
                pygame.mixer.music.load("p2.mp3")
                pygame.mixer.music.play()

            elif self.e1 == 'Derecha' and self.e2 == 'Derecha':
                pygame.mixer.init()
                pygame.mixer.music.load("p3.mp3")
                pygame.mixer.music.play()

            self.count = 0
            self.e1 = 'Nada'
            self.e2 = 'Nada'

        if self.count == 10:

            if self.e1 != 'Nada':

                if self.e1 == 'Parpadeo':

                    pygame.mixer.init()
                    pygame.mixer.music.load("p4.mp3")
                    pygame.mixer.music.play()

                elif self.e1 == 'Derecha':
                    pygame.mixer.init()
                    pygame.mixer.music.load("p5.mp3")
                    pygame.mixer.music.play()

                elif self.e1 == 'Izquierda':
                    pygame.mixer.init()
                    pygame.mixer.music.load("p6.mp3")
                    pygame.mixer.music.play()


            self.count = 0
            self.e1 = 'Nada'
            self.e2 = 'Nada'

class proc_wind:
    def __init__(self, channels, window, act):
        self.data = np.zeros((window, channels))
        self.inc_lenght = act

    def refresh(self, inc_data):
        ind = self.inc_lenght
        temp = self.data[ind:, :]
        self.data[:-ind, :] = temp
        self.data[-ind:, :] = np.array(inc_data)

