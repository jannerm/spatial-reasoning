spritepath = 'sprites/'

objects = {
    'grass': {
        'index': 0,
        'value': 0,
        'sprite': 'sprites/grass_figure_4.png', # 'sprites/white.png',
        'background': True,
        'unique': False,
        },
    'puddle': {
        'index': 1,
        'value': -1,
        'sprite': 'sprites/water_figure_2.png', 
        'background': True,
        'unique': False,
        },
    ## unique
    'star': {
        'index': 2, 
        'value': 0,
        'sprite': 'sprites/star_figure-01.png', ## white_alpha.png
        'background': False,
        'unique': True,
        },
    'circle': {
        'index': 3, 
        'value': 0,
        'sprite': 'sprites/circle_figure-01.png',
        'background': False,
        'unique': True,
        },
    'triangle': {
        'index': 4,
        'value': 0,
        'sprite': 'sprites/triangle_figure-01.png',
        'background': False,
        'unique': True,
        },
    'heart': {
        'index': 5,
        'value': 0,
        'sprite': 'sprites/heart_figure-01.png',
        'background': False,
        'unique': True,
        },
    'spade': {
        'index': 6,
        'value': 0,
        'sprite': 'sprites/spade_figure-01.png',
        'background': False,
        'unique': True,
        },
    'diamond': {
        'index': 7,
        'value': 0,
        'sprite': 'sprites/diamond_figure-01.png',
        'background': False,
        'unique': True,
        },
    ## non-unique
    'rock': {
        'index': 8,
        'value': 0,
        'sprite': 'sprites/rock_figure-01.png',
        'background': False,
        'unique': False,
        },
    'tree': {
        'index': 9,
        'value': 0,
        'sprite': 'sprites/tree_figure-01.png',
        'background': False,
        'unique': False,
        },
    'house': {
        'index': 10,
        'value': 0,
        'sprite': 'sprites/house_figure-01.png',
        'background': False,
        'unique': False,
    },
    'horse': {
        'index': 11,
        'value': 0,
        'sprite': 'sprites/horse_figure-01.png',
        'background': False,
        'unique': False,
    },
}

unique_instructions = {
    ## original 
    'to top left of': (-1, -1),
    'on top of': (-1, 0),
    'to top right of': (-1, 1),
    'to left of': (0, -1),
    'with': (0, 0),
    'to right of': (0, 1),
    'to bottom left of': (1, -1),
    'on bottom of': (1, 0),
    'to bottom right of': (1, 1),
    ## two steps away
    'two to the left and two above': (-2, -2),
    'one to the left and two above': (-2, -1),
    'two above': (-2, 0),
    'one to the right and two above': (-2, 1),
    'two to the right and two above': (-2, 2),
    'two to the right and one above': (-1, 2),
    'two to the right of': (0, 2),
    'two to the right and one below': (1, 2),
    'two to the right and two below': (2, 2),
    'one to the right and two below': (2, 1),
    'two below': (2, 0),
    'one to the left and two below': (2, -1),
    'two to the left and two below': (2, -2),
    'two to the left and one below': (1, -2),
    'two to the left': (0, -2),
    'two to the left and one above': (-1, -2) 

}

background = 'sprites/grass_figure_4.png'

# print objects
