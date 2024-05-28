import pygame
import math
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Toplevel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def main():
    pygame.init()
    pygame.display.set_caption("testowa aplikacja")

    # screen settings
    WIDTH = 1200
    HEIGHT = 900
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    marginX = 100
    marginY = 100

    # clock settings
    timer = pygame.time.Clock()
    fps = 60

    # cursor settings
    cursorArrow = pygame.cursors.Cursor(pygame.SYSTEM_CURSOR_ARROW)
    cursorHand = pygame.cursors.Cursor(pygame.SYSTEM_CURSOR_HAND)
    pygame.mouse.set_cursor(cursorArrow)

    # arc center
    arcCenter = ((WIDTH) / 2, (HEIGHT) / 2 + marginY)

    # compass settings
    compassImg = pygame.image.load("compass.png")
    compassImg = pygame.transform.scale(compassImg, (300, 300))
    compassRect = compassImg.get_rect(center=arcCenter)

    # speaker settings
    speakerDefaultImg = pygame.image.load("speaker_no_sound.png")
    speakerDefaultImg = pygame.transform.scale(speakerDefaultImg, (100, 100))
    speakerAngle = 0
    speakerSoundImg = pygame.image.load("speaker.png")
    speakerSoundImg = pygame.transform.scale(speakerSoundImg, (100, 100))
    speakerImg = speakerDefaultImg
    speakerRect = speakerImg.get_rect(center=(arcCenter[0], marginY + 15))

    def draw_arc():
        pygame.draw.arc(screen, 'black', (marginX, marginY, WIDTH - 2 * marginX, HEIGHT), 0, math.pi, 3)

    def draw_speaker():
        screen.blit(speakerDefaultImg, speakerRect)

    def rotate(image, angle, x, y):
        rotated_image = pygame.transform.rotate(image, angle)
        new_rect = rotated_image.get_rect(center=image.get_rect(center=(x, y)).center)
        return rotated_image, new_rect

    def draw_sine_wave():
        root = tk.Tk()
        root.title("Wykres Fali Sinusoidalnej")

        fig, ax = plt.subplots()
        t = np.linspace(0, 1, 400)
        y = np.sin(2 * np.pi * 10 * t)
        ax.plot(t, y)

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack()

        root.mainloop()

    class Sound():
        def __init__(self):
            pygame.mixer.init(size=32)
            self.samplingRate = 44100
            self.frequency = 440.0

        def generateSound(self):
            buffer = (np.sin(2 * np.pi * np.arange(self.samplingRate) * self.frequency / self.samplingRate)).astype(np.float32)
            sound = pygame.mixer.Sound(buffer)
            return buffer

    # main loop parameters
    clicked = False
    running = True
    angle = 0
    moving = False
    sound = Sound()

    # main loop
    while running:
        timer.tick(fps)
        screen.fill('grey')

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if speakerRect.collidepoint(event.pos):
                    pygame.mouse.set_cursor(cursorHand)
                    clicked = True

            elif event.type == pygame.MOUSEBUTTONUP:
                pygame.mouse.set_cursor(cursorArrow)
                if not moving and clicked:
                    sound.generateSound()
                    draw_sine_wave()
                    if speakerImg == speakerSoundImg:
                        speakerImg = speakerDefaultImg
                    else:
                        speakerImg = speakerSoundImg
                elif clicked:
                    speakerImg = speakerDefaultImg
                moving = False
                clicked = False

            elif event.type == pygame.MOUSEMOTION and clicked:
                # move with the mouse
                speakerRect[0] += event.rel[0]
                speakerRect[1] += event.rel[1]
                moving = True
                speakerImg = speakerDefaultImg

                # calculate distances
                xDist = event.pos[0] - arcCenter[0]
                yDist = arcCenter[1] - event.pos[1]
                angle = math.degrees(math.atan2(yDist, xDist)) - 90

        draw_arc()
        rotated_image, new_rect = rotate(speakerImg, angle, speakerRect.center[0], speakerRect.center[1])
        speakerRect = new_rect
        screen.blit(rotated_image, speakerRect)
        pygame.draw.rect(screen, "red", (arcCenter[0], arcCenter[1], 10, 10))
        screen.blit(compassImg, compassRect)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
