#Class Practice - EX45

#this application will mimick a concert where three different bands play a set of songs each
#each band is different in terms of music style, etc
#the application will manage each of the three phases of the concert and run them sequentially


class Concert(object):

    def __init__(self):
        self.numberOfPhases = 2
        self.numberOfBands = 2
        self.minutesAllowedForEachBand = 100

    def start(self):
        print("The concert is starting right now")
        print(f"The concert will last {self.numberOfBands*self.minutesAllowedForEachBand} minutes")

    def end(self):
        print("That was the end of the concert")
        print("I hope you enjoyed it")
        print("How much did you like it, from 1 to 10?")
        rating = int(input("> "))
        print("You gave the concert a: ", rating)

        if rating > 7:
            print("We are glad you liked it!")
        elif rating > 5:
            print("We are glad you liked it at least a little bit")
        else:
            print("We are sorry that you didn\'t like it at all")

class Band(object):

    def __init__(self):
        self.numberOfMembers = numberOfMembers
        self.styleOfMusic = styleOfMusic
        self.numberOfFans = numberOfFans

    def playMusic(self):
        pass

    def scream(self):
        pass

#what if I wanted to pass the variables when instantiating the object rather than fixing the values now?
class RockBand(Band):

    #how do i pass a variable from the Concert class such as concert duration?
    #should I make this class a child classof concert or should I do it differently?
    def __init__(self):
        super(Band, self).__init__()
        self.mood = 'Relaxed'


    def playMusic(self, mood):
        if mood == 'Relaxed':
            count = 0
            while count < 100:
                print("---"*4)
                print("LaLaLa"*10)
                print("RoRo"*3)
                print("Soooooooooo"*2)
                count+=1
        elif mood == 'Angry':
            count = 0
            while count < 100:
                print("---"*4)
                print("RARAR!!!!!ARARAR!!!"*10)
                print("YOYAOT!!"*10)
                print("01010101!!!")
                print("HELLYAAAAA!!!!!!"*2)
                count+=1
        else:
            exit(0)

class ElectroBand(Band):

    def __init__(self):
        super(Band, self).__init__()
        self.beat = 'Fast'

    def playMusic(self, beat):
                if beat == 'Fast':
                    count = 0
                    while count < 100:
                        print("---"*4)
                        print("tuz-tuz-tuz"*10)
                        print("tadatuz"*3)
                        print("brbrbrbr"*2)
                        count+=1
                elif beat == 'Slow':
                    count = 0
                    while count < 100:
                        print("---"*4)
                        print("tuuuuztuuuuz"*10)
                        print("dadadadada!!"*10)
                        print("tuuuuztuuuuz!!!")
                        print("tuuuuztuuuuz!!!!!!"*2)
                        count+=1
                else:
                    exit(0)

class ManageConcert(object):

    def __init__(self):
        self.concert = Concert()
        self.rockband = RockBand()
        self.electroband = ElectroBand()

    def introShow(self):
        print("Introduction Presentation with Cool Dude")
        for i in range(10):
            print("--------"*10 + "\n")

    def takeBreak(self):
        print("Let's take a break")
        for i in range(10):
            print("[][][][][][][][]"*10 + "\n")

    def runConcert(self):
        self.concert.start()
        self.introShow()
        self.rockband.playMusic("Relaxed")
        self.takeBreak()
        self.electroband.playMusic("Fast")
        self.concert.end()

theConcert = ManageConcert()
theConcert.runConcert()
