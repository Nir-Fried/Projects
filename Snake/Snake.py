import pygame
import random

pygame.init()
width=800
height=600
screen = pygame.display.set_mode((width,height))

pygame.display.set_caption('Snake')
blue = (0,0,255)
red = (255,0,0)
green = (0,255,0)
white = (255,255,255)
black=(0,0,0)
clock = pygame.time.Clock()
smallfont = pygame.font.SysFont(None,25)
mediumfont= pygame.font.SysFont(None,50)
bigfont = pygame.font.SysFont(None,80)

def score(score):
    text = smallfont.render("Score: "+str(score),True, white )
    screen.blit(text,[0,0])

def pause():
    msg("Paused",white,-100,size="big")
    msg("Press Escape to continue or N to quit",white,25,size ="medium")
    pygame.display.update()
    paused = True
    while paused:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    paused = False
                elif event.key == pygame.K_n:
                    pygame.quit()
                    quit()
            clock.tick(10)
def openscreen():
    starting = True
    while starting:
        screen.fill(black)
        msg("Snake",green,-100,size="big")
        msg("Press 'Y' to start playing or 'N' to quit ",white,size ="medium")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:
                    pygame.quit()
                    quit()
                if event.key == pygame.K_y:
                    starting = False

        pygame.display.update()
        clock.tick(30)

def snake(lenlist):
    for xy in lenlist:
        pygame.draw.rect(screen,green, pygame.Rect(xy[0], xy[1], 10, 10))

def sizer(text,color,size):
    if size == "small":
        textsurf = smallfont.render(text,True,color)
    if size == "medium":
        textsurf = mediumfont.render(text,True,color)
    if size == "big":
        textsurf =bigfont.render(text,True,color)
    return textsurf,textsurf.get_rect()

def msg(msg,color,mover=0,size = "small"):
    textsurf,textshape = sizer(msg,color,size)
    textshape.center = (width/2), (height/2)+mover
    screen.blit(textsurf,textshape)

def RunGame():
    done = False
    replay = False
    x = width/2
    y = height/2
    movesize =10
    xmove =0
    ymove=0
    lastpressed = None
    lenlist = []
    length = 1

    AppleX= round(random.randrange (0,width-10)/10.0)*10.0
    AppleY= round (random.randrange (0,height-10)/10.0)*10.0
    while not done:
        if replay == True:
            msg ("Game Over", red ,-100,size = "big")
            msg("Press 'Y' to play again or 'N' to quit ",white,size ="medium")
            pygame.display.update()
        while replay:
            screen.fill((0,0,0))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    replay = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_n:
                        done = True
                        replay = False
                    if event.key == pygame.K_y:
                        RunGame()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                    done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and lastpressed != "r":
                    xmove=-movesize
                    ymove=0
                    lastpressed = "l"
                elif event.key == pygame.K_RIGHT and lastpressed != "l":
                    xmove=movesize
                    ymove= 0
                    lastpressed = "r"
                elif event.key == pygame.K_UP and lastpressed != "d":
                    ymove=-movesize
                    xmove=0
                    lastpressed = "u"
                elif event.key == pygame.K_DOWN and lastpressed != "u":
                    ymove=movesize
                    xmove= 0
                    lastpressed="d"
                elif event.key == pygame.K_ESCAPE:
                    pause()
        if x < 0 or x>=width or y<0 or y>=height:
            replay =  True

        x=x+xmove
        y=y+ymove
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen,red,(AppleX,AppleY,10,10))

        head = []
        head.append(x)
        head.append(y)
        lenlist.append(head)
        if len(lenlist) > length:
            del lenlist[0]


        for cord in lenlist[:-1]:
            if cord == head:
                replay= True

        score(length-1)

        snake(lenlist)
        pygame.display.update()
        clock.tick(20)

        if x==AppleX and y==AppleY:
            AppleX= round(random.randrange (0,width-10)/10.0)*10.0
            AppleY= round (random.randrange (0,height-10)/10.0)*10.0
            length=length+3

    pygame.quit()
    quit()
openscreen()
RunGame()
