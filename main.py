import pygame
import math
import pyaudio
import numpy as np
import tkinter as tk
from tkinter import Toplevel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

# help(pygame)

class Sound():
    def __init__(self, frequency=440.0, samplingRate=44100):
        pygame.mixer.init(size=32)
        self.samplingRate = samplingRate
        self.frequency = frequency

    # jak chce z odsluchem to raczej inny watek bo zatrzymuje petle
    def generate_sound(self):
        buffer = (np.sin(2 * np.pi * np.arange(self.frequency) * self.frequency / self.samplingRate)).astype(
            np.float32)
        sound = pygame.mixer.Sound(buffer)
        # sound.play()
        # sound.stop()
        # pygame.mixer.quit()
        # sound.play()
        # pygame.time.wait(int(sound.get_length() * 1000))
        return buffer

    def generate_sound_pyaudio(self):
        p = pyaudio.PyAudio()
        volume = 0.5
        samplingRate = 44100
        freq = 440.0

        samples = (np.sin(2 * np.pi * np.arange(samplingRate) * freq / samplingRate)).astype(np.float32).tobytes()

        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=samplingRate,
                        output=True)

        stream.write(samples)
        stream.stop_stream()
        stream.close()
        p.terminate()

class Plotter():
    @staticmethod
    def draw(x=None):
        def _draw():
            root = tk.Tk()
            root.title("Wykres fali")

            fig, ax = plt.subplots()
            if x is None:
                xData = np.linspace(0, 1, 440)
            else:
                xData = x
            y = np.sin(2 * np.pi * 5 * xData)
            ax.plot(xData, y)

            canvas = FigureCanvasTkAgg(fig, master=root)
            canvas.draw()
            canvas.get_tk_widget().pack()

            root.mainloop()

        plotThread = threading.Thread(target=_draw)
        plotThread.start()

class Button():
    def __init__(self, x, y, image):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.topleft = (x,y)

    def draw(self, screen):
        screen.blit(self.image, (self.rect.x, self.rect.y))

class PygameApp():
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("testowa aplikacja")

        # screen settings
        self.WIDTH = 1200
        self.HEIGHT = 900
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.marginX = 100
        self.marginY = 100

        # clock settings
        self.timer = pygame.time.Clock()
        self.fps = 60

        # cursor settings
        self.cursorArrow = pygame.cursors.Cursor(pygame.SYSTEM_CURSOR_ARROW)
        self.cursorHand = pygame.cursors.Cursor(pygame.SYSTEM_CURSOR_HAND)
        pygame.mouse.set_cursor(self.cursorArrow)

        # arc center
        self.arcCenter = ((self.WIDTH) / 2, (self.HEIGHT) / 2 + self.marginY)

        # compass settings
        self.compassImg = pygame.image.load("compass.png")
        self.compassImg = pygame.transform.scale(self.compassImg, (300, 300))
        self.compassRect = self.compassImg.get_rect(center=self.arcCenter)
        self.compass_rotated_image = self.compassImg

        # speaker settings
        self.speakerDefaultImg = pygame.image.load("speaker_no_sound.png")
        self.speakerDefaultImg = pygame.transform.scale(self.speakerDefaultImg, (100, 100))
        self.speakerAngle = 0
        self.speakerSoundImg = pygame.image.load("speaker.png")
        self.speakerSoundImg = pygame.transform.scale(self.speakerSoundImg, (100, 100))
        self.speakerImg = self.speakerDefaultImg
        self.speakerRect = self.speakerImg.get_rect(center=(self.arcCenter[0], self.marginY + 15))

        # icons
        self.eyeIcon = pygame.image.load("icon.png").convert_alpha()
        self.eyeIcon = pygame.transform.scale(self.eyeIcon, (30,30))

        # main loop parameters
        self.clicked = False
        self.clickedEye = False
        self.running = True
        self.angle = 0
        self.moving = False
        self.sound = Sound()
        self.plotter = Plotter()
        self.wave = None

        self.eyeButton = Button(10 + 140, self.arcCenter[1] + self.marginY + 60, self.eyeIcon)



    def draw_arc(self):
        pygame.draw.arc(self.screen, 'black', (self.marginX, self.marginY, self.WIDTH - 2 * self.marginX, self.HEIGHT), 0, math.pi, 3)

    def draw_speaker(self):
        self.screen.blit(self.speakerDefaultImg, self.speakerRect)

    def rotate(self, image, angle, x, y):
        rotated_image = pygame.transform.rotate(image, angle)
        new_rect = rotated_image.get_rect(center=image.get_rect(center=(x, y)).center)

        return rotated_image, new_rect

    def display_text(self, text, marginY, startPos=10):
        myFont = pygame.font.SysFont("Comic Sans MS", 20)
        textSurface = myFont.render(text, False, (0, 0, 0))
        self.screen.blit(textSurface, (startPos, self.arcCenter[1] + self.marginY + marginY))

    def print_details(self):
        pygame.font.init()
        #self.display_text("Speaker position: " + "x: " + str(self.speakerRect[0]) + "y: " + str(self.speakerRect[1]), 0)
        self.display_text("Speaker position: " + str(self.speakerRect), 0)
        self.display_text("Angle: ", 30)
        self.display_text("Wave details: ", 60)
        # tutaj przycisk
        self.eyeButton.draw(self.screen)

        self.display_text("Speaker angle: " + str(self.speakerAngle), 90)
        # ogarnac tu
        self.display_text("Compass angle: " + str(self.angle), 120)


    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.speakerRect.collidepoint(event.pos):
                    pygame.mouse.set_cursor(self.cursorHand)
                    self.clicked = True
                elif self.eyeButton.rect.collidepoint(event.pos):
                    pygame.mouse.set_cursor(self.cursorHand)
                    self.clickedEye = True

            elif event.type == pygame.MOUSEBUTTONUP:
                pygame.mouse.set_cursor(self.cursorArrow)
                if not self.moving and self.clicked:
                    # generowac tylko po odpaleniu
                    self.wave = self.sound.generate_sound()

                    # speakerAngle na angle
                    self.compass_rotated_image, new_rect = self.rotate(self.compassImg, self.speakerAngle, self.compassRect.center[0],
                                                          self.compassRect.center[1])
                    self.compassRect = new_rect
                    self.angle = self.speakerAngle

                    if self.speakerImg == self.speakerSoundImg:
                        self.speakerImg = self.speakerDefaultImg
                    else:
                        self.speakerImg = self.speakerSoundImg
                elif self.clicked:
                    self.speakerImg = self.speakerDefaultImg
                elif self.clickedEye:
                    # tutaj cos ogarnac
                    self.plotter.draw(self.wave)
                self.moving = False
                self.clicked = False

            elif event.type == pygame.MOUSEMOTION and self.clicked:
                # move with the mouse
                self.speakerRect[0] += event.rel[0]
                self.speakerRect[1] += event.rel[1]
                self.moving = True
                self.speakerImg = self.speakerDefaultImg

                # calculate distances
                # TODO odleglosc od srodka rect a nie od pozycji kursora
                xDist = event.pos[0] - self.arcCenter[0]
                yDist = self.arcCenter[1] - event.pos[1]
                self.speakerAngle = math.degrees(math.atan2(yDist, xDist)) - 90

    def run(self):
        # main loop
        while self.running:
            self.timer.tick(self.fps)
            self.screen.fill('white')
            self.print_details()

            self.handle_events()

            self.draw_arc()
            rotated_image, new_rect = self.rotate(self.speakerImg, self.speakerAngle, self.speakerRect.center[0], self.speakerRect.center[1])
            self.speakerRect = new_rect
            self.screen.blit(rotated_image, self.speakerRect)
            pygame.draw.rect(self.screen, "red", (self.arcCenter[0], self.arcCenter[1], 10, 10))
            self.screen.blit(self.compass_rotated_image, self.compassRect)
            pygame.display.flip()

        pygame.quit()

if __name__ == "__main__":
    app = PygameApp()
    app.run()