import sys

import pygame
from pygame import QUIT


def pygame_events(space, myenv, height):
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONUP:
            x, y = pygame.mouse.get_pos()
            myenv.change_target_pos(x, height - y)