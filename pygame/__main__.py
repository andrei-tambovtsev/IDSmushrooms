import time

import numpy
import pygame  # python3 -m pip install pygame
import pygame_widgets  # python3 -m pip install pygame_widgets
import pickle

# This file contains some pygame-related logic. For code that actually predicts, go to run_end

screen = ''
data = [0] * 14  # Storing data
decision = -1  # 1 = edible, 0 = poisonous
stage = 0

model = pickle.load(open('dt-model.pickle', 'rb'))

latest_click = 0  # Accidental click prevention

ranges = [0, 0, 8, 10, 15]
rf = [0,1,2,3,4,5,6,7,8,-1,9,10,11,12,13]

def prepare():
    global screen
    screen.fill((255, 255, 255))
    screen.blit(background_image, (0, 0))
    for c in [(10, 400), (140, 300), (250, 330), (400, 275)]:
        screen.blit(mushroom_image, c)
    # pygame.draw.circle(screen, (0, 0, 255), (250, 250), 75)


def run_forward(index):
    global data, stage, latest_click

    if latest_click + 0.33 > time.time():  # Accidental click prevention, usability in mind!
        return
    latest_click = time.time()

    if index == -1 and stage == 0:
        stage = 1
        return

    if (rf[index] != -1):
        data[rf[index]] = 1
    stage += 1
    if stage == 4:
        run_end()


def run_end():
    # Actual ML happens here
    global stage, data, decision

    temp = numpy.array(data).reshape(1, -1)
    dec = model.predict_proba(temp)[:, 1]

    decision = 1 if dec >= 0.5 else 0

    # Cleanup
    stage = 0
    data = [0] * 14


if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption('Shroomer 1 - A4 edition')
    screen = pygame.display.set_mode([500, 500])

    mushroom_image = pygame.image.load('./img/mushroom.png')
    mushroom_image = pygame.transform.scale(mushroom_image, (100, 100))
    background_image = pygame.image.load('./img/background.png')
    background_image = pygame.transform.scale(background_image, (500, 500))
    poisonous_image = pygame.image.load('./img/poisonous.jpg')
    poisonous_image = pygame.transform.scale(poisonous_image, (200, 100))
    edible_image = pygame.image.load('./img/edible.jpg')
    edible_image = pygame.transform.scale(edible_image, (200, 100))

    running = True

    # gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
    # stalk-shape: enlarging=e,tapering=t
    # stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?

    # 'gill-color_b', 'gill-color_g', 'gill-color_h', 'gill-color_k', 'gill-color_n', 'gill-color_p', 'gill-color_u', 'gill-color_w',
    # 'stalk-shape_e',
    # 'stalk-root_?', 'stalk-root_b', 'stalk-root_c', 'stalk-root_e', 'stalk-root_r'

    titles = ['', 'Gill color?', 'Is stalk enlarging?', 'Characterize stalk root']

    texts = ['Buff', 'Gray', 'Chocolate', 'Black', 'Brown', 'Pink', 'Purple', 'White',
             'Yes', 'No',
             'No stalk root', 'Bulbous', 'Club', 'Equal', 'Rooted']

    start_button = pygame_widgets.Button(
        screen, 100, 10, 300, 40, text="Start",
        inactiveColour=(196,128,128), radius=4, fontSize=16, margin=20,
        onClick=lambda v=-1: run_forward(v)
    )

    choose_buttons = []
    for i in range(len(texts)):
        pospos = i
        if pospos >= 10:
            pospos -= 10
        if pospos >= 8:
            pospos -= 8
        button = pygame_widgets.Button(
            screen, 50, 75 + pospos * 50, 200, 40, text=texts[i],
            inactiveColour=(196, 128, 128), radius=4, fontSize=16, margin=20,
            onClick=lambda v=i: run_forward(v)
        )
        choose_buttons += [button]

    while running:
        events = pygame.event.get()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
                quit()

        prepare()

        myfont = pygame.font.SysFont('Comic Sans MS', 30)
        textsurface = myfont.render(titles[stage], False, (255, 128, 128))
        screen.blit(textsurface, (50, 20))

        if stage == 0:
            start_button.listen(events)
            start_button.draw()
            if decision == 1:
                screen.blit(edible_image, (150, 100))
            if decision == 0:
                screen.blit(poisonous_image, (150, 100))

        for i in range(ranges[stage], ranges[stage+1]):
            choose_buttons[i].draw()
            choose_buttons[i].listen(events)

        pygame.display.flip()
    pygame.quit()
