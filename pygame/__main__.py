import pygame  # python3 -m pip install pygame
import pygame_widgets  # python3 -m pip install pygame_widgets

screen = ''
data = [0] * 18
decision = -1
stage = 0

def prepare():
    global screen
    screen.fill((255, 255, 255))
    screen.blit(background_image, (0, 0))
    for c in [(10, 400), (140, 300), (250, 330), (400, 275)]:
        screen.blit(mushroom_image, c)
    # pygame.draw.circle(screen, (0, 0, 255), (250, 250), 75)


def run_forward(number):
    global data, stage
    if (number != -1):
        data[number] = 1
    stage += 1

    if stage == 4:
        run_end()


def run_end():
    # Actual ML here
    global stage, data, decision
    stage = 0

if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption('Shroomer 1 - A4 edition')
    screen = pygame.display.set_mode([500, 500])

    mushroom_image = pygame.image.load('./img/mushroom.png')
    mushroom_image = pygame.transform.scale(mushroom_image, (100, 100))
    background_image = pygame.image.load('./img/background.png')
    background_image = pygame.transform.scale(background_image, (500, 500))

    running = True

    # gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
    # stalk-shape: enlarging=e,tapering=t
    # stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?

    # 'gill-color_b', 'gill-color_g', 'gill-color_h', 'gill-color_k', 'gill-color_n', 'gill-color_p', 'gill-color_u', 'gill-color_w',
    # 'stalk-shape_e',
    # 'stalk-root_?', 'stalk-root_b', 'stalk-root_c', 'stalk-root_e', 'stalk-root_r'

    titles = ['Gill color?', 'Is stalk enlarging?', 'Characterize stalk root']

    texts = ['Buff', 'Gray', 'Chocolate', 'Black', 'Brown', 'Pink', 'Purple', 'White',
             'Yes', 'No',
             'No stalk root', 'Bulbous', 'Club', 'Equal', 'Rooted']

    start_button = pygame_widgets.Button(
        screen, 100, 10, 300, 40, text="Start", inactiveColour=(196,128,128),
        radius=4, fontSize=16, margin=20,
        onClick=lambda v=-1: run_forward(v)
    )

    choose_buttons = []
    for i in range(len(texts)):
        button = pygame_widgets.Button(
            screen, 50, 75 + i * 50, 200, 40, text=texts[i],
            fontSize=16, margin=20,
            inactiveColour=(128, 128, 128),
            pressedColour=(0, 255, 0), radius=4,
            onClick=lambda: print('Click')
        )
        button.draw()

    while running:
        events = pygame.event.get()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
                quit()

        prepare()
        if stage == 0:
            if decision == 0:
                pass
            start_button.draw()
        if stage == 1:
            pass
        if stage == 2:
            pass
        if stage == 3:
            pass

        start_button.listen(events)
        for butt in choose_buttons:
            butt.listen(events)
        pygame.display.flip()
    pygame.quit()