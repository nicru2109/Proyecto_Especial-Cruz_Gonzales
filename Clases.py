## Imports
import time
import numpy as np
import pygame
import pandas as pd

## Clases
class pre_wind:

    def __init__(self):
        self.data = []
        self.count = 0
        self.array = []
        self.e1 = 'Nada'
        self.e2 = 'Nada'

    def add_null(self):
        self.array.append('Nada')

    def actualizar(self, Var = 'Nada', mode='train', sig_type='EOG'):
        self.count += 1
        if Var != 'Nada':
            if self.e1 == 'Nada':
                self.e1 = Var
                self.count = 0
            else:
                self.e2 = Var

        if self.e2 != 'Nada':

            if self.e1 == 'P' and self.e2 == 'P':
                # pygame.mixer.init()
                # pygame.mixer.music.load("mp3//p1.mp3")
                # pygame.mixer.music.play()
                # time.sleep(2)
                # pygame.mixer.music.stop()
                print('PP')
                if mode == 'test':
                    self.array.append('PP')


            elif self.e1 == 'P' and self.e2 == 'I':
                # pygame.mixer.init()
                # pygame.mixer.music.load("mp3//p2.mp3")
                # pygame.mixer.music.play()
                # time.sleep(2)
                # pygame.mixer.music.stop()
                print('PI')

                if mode == 'test':
                    self.array.append('PI')

            elif self.e1 == 'P' and self.e2 == 'D':
                # pygame.mixer.init()
                # pygame.mixer.music.load("mp3//p3.mp3")
                # pygame.mixer.music.play()
                # time.sleep(2)
                # pygame.mixer.music.stop()
                print('PD')

                if mode == 'test':
                    self.array.append('PD')

            elif self.e1 == 'I' and self.e2 == 'P':
                # pygame.mixer.init()
                # pygame.mixer.music.load("mp3//p4.mp3")
                # pygame.mixer.music.play()
                # time.sleep(2)
                # pygame.mixer.music.stop()
                print('IP')

                if mode == 'test':
                    self.array.append('IP')

            elif self.e1 == 'D' and self.e2 == 'P':
                # pygame.mixer.init()
                # pygame.mixer.music.load("mp3//p5.mp3")
                # pygame.mixer.music.play()
                # time.sleep(2)
                # pygame.mixer.music.stop()
                print('DP')

                if mode == 'test':
                    self.array.append('DP')

            elif self.e1 == 'MF' and self.e2 == 'MF':
                # pygame.mixer.init()
                # pygame.mixer.music.load("mp3//p6.mp3")
                # pygame.mixer.music.play()
                # time.sleep(2)
                # pygame.mixer.music.stop()

                if mode == 'test':
                    self.array.append('MF')

            elif self.e1 == 'CD' and self.e2 == 'CD':
                # pygame.mixer.init()
                # pygame.mixer.music.load("mp3//p7.mp3")
                # pygame.mixer.music.play()
                # time.sleep(2)
                # pygame.mixer.music.stop()

                if mode == 'test':
                    self.array.append('CD')


            elif self.e1 == 'CI' and self.e2 == 'CI':
                # pygame.mixer.init()
                # pygame.mixer.music.load("mp3//p8.mp3")
                # pygame.mixer.music.play()
                # time.sleep(2)
                # pygame.mixer.music.stop()

                if mode == 'test':
                    self.array.append('CI')

            elif self.e1 == 'C' and self.e2 == 'C':
                # pygame.mixer.init()
                # pygame.mixer.music.load("mp3//p9.mp3")
                # pygame.mixer.music.play()
                # time.sleep(2)
                # pygame.mixer.music.stop()

                if mode == 'test':
                    self.array.append('C')

            self.count = 0
            self.e1 = 'Nada'
            self.e2 = 'Nada'

        if self.count == 10:

            if self.e1 != 'Nada':

                if self.e1 == 'P':

                    # pygame.mixer.init()
                    # pygame.mixer.music.load("mp3//p10.mp3")
                    # pygame.mixer.music.play()
                    # time.sleep(2)
                    # pygame.mixer.music.stop()
                    print('P')

                    if mode == 'test':
                        self.array.append('P')

                elif self.e1 == 'D':
                    # pygame.mixer.init()
                    # pygame.mixer.music.load("mp3//p11.mp3")
                    # pygame.mixer.music.play()
                    # time.sleep(2)
                    # pygame.mixer.music.stop()
                    print('D')

                    if mode == 'test':
                        self.array.append('D')

                elif self.e1 == 'I':
                    # pygame.mixer.init()
                    # pygame.mixer.music.load("mp3//p12.mp3")
                    # pygame.mixer.music.play()
                    # time.sleep(2)
                    # pygame.mixer.music.stop()
                    print('I')

                    if mode == 'test':
                        self.array.append('I')

                # elif self.e1 == 'MF':
                #     pygame.mixer.init()
                #     pygame.mixer.music.load("mp3//p6.mp3")
                #     pygame.mixer.music.play()
                #     time.sleep(2)
                #     pygame.mixer.music.stop()
                #
                #     if mode == 'test':
                #         self.array.append('MF')
                #
                # elif self.e1 == 'CD':
                #     pygame.mixer.init()
                #     pygame.mixer.music.load("mp3//p7.mp3")
                #     pygame.mixer.music.play()
                #     time.sleep(2)
                #     pygame.mixer.music.stop()
                #
                #     if mode == 'test':
                #         self.array.append('CD')
                #
                # elif self.e1 == 'CI':
                #     pygame.mixer.init()
                #     pygame.mixer.music.load("mp3//p8.mp3")
                #     pygame.mixer.music.play()
                #     time.sleep(2)
                #     pygame.mixer.music.stop()
                #
                #     if mode == 'test':
                #         self.array.append('CI')
                #
                # elif self.e1 == 'C':
                #     pygame.mixer.init()
                #     pygame.mixer.music.load("mp3//p9.mp3")
                #     pygame.mixer.music.play()
                #     time.sleep(2)
                #     pygame.mixer.music.stop()
                #
                #     if mode == 'test':
                #         self.array.append('C')

            self.count = 0
            self.e1 = 'Nada'
            self.e2 = 'Nada'

            print('Nada')


class proc_wind:
    def __init__(self, channels, window, act):
        self.data = np.zeros((window, channels))
        self.inc_lenght = act

    def refresh(self, inc_data):
        ind = self.inc_lenght
        temp = self.data[ind:, :]
        self.data[:-ind, :] = temp
        self.data[-ind:, :] = np.array(inc_data)

class sig_sym:
    def __init__(self, path, sRate):

        Temp_read1 = pd.read_csv(path, header=4)
        self.data = Temp_read1[[' EXG Channel 0', ' EXG Channel 1', ' EXG Channel 2', ' EXG Channel 3', ' EXG Channel 4', ' EXG Channel 5', ' EXG Channel 6', ' EXG Channel 7']].to_numpy()

        self.delay = 1 / sRate
        self.len = len(self.data)
        self.count = 0

        if self.count < self.len:
            self.is_left = True
    def get(self):

        time.sleep(self.delay)

        temp = self.data[self.count, :]

        self.count += 1

        if self.count == self.len:
            self.is_left = False

        return temp

    def reset(self):
        self.count = 0
