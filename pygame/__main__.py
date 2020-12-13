import pygame  # python3 -m pip install pygame

screen = ''

def prepare():
    screen.fill((255, 255, 255))
    screen.blit(background_image, (0, 0))
    for c in [(10, 400), (140, 300), (250, 330), (400, 275)]:
        screen.blit(mushroom_image, c)
    # pygame.draw.circle(screen, (0, 0, 255), (250, 250), 75)

if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption('Shroomer 1 - A4 edition')
    screen = pygame.display.set_mode([500, 500])

    mushroom_image = pygame.image.load('./img/mushroom.png')
    mushroom_image = pygame.transform.scale(mushroom_image, (100, 100))
    background_image = pygame.image.load('./img/background.png')
    background_image = pygame.transform.scale(background_image, (500, 500))

    running = True

    data
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        prepare()

        pygame.display.flip()
    pygame.quit()