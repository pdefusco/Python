from nose.tools import *
#from proj2.lexicon import Lexicon
import proj2

def test_directions():

    lexicon = Lexicon()

    assert_equal(lexicon.scan("north"), [('direction', 'north')])
    result = lexicon.scan("north south east")
    assert_equal(result, [('direction','north'),('direction','south'),('direction','east')])









'''def setup():
    print('SETUP!')

def teardown():
    print('TEAR DOWN!')

def test_basic():
    print("I RAN!")
'''
