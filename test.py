import time
import os
import random
import tkinter as tk
import threading
import curses
def keyloop(flapper, ai):
    def update(c, stdscr):
        result = ["", [0]]
        if c%2==0:
            result = flapper.update_objects()
            if result[0] > 0:
                ai.reward(result[1], result[0])
            elif result[0] < 0:
                ai.punish(result[1], result[0])
            if round(ai.feedforward(0,result[1])[0]) >= 1:
                flapper.flap()
        if c%16==0:
            flapper.make_pipe()
        return(c+1)
    def main(stdscr):
        curses.curs_set(0)
        curses.noecho()
        stdscr.nodelay(True)
        stdscr.clear()
        counter = 0
        while True:
            counter = update(counter, stdscr)
            if counter > 50000:
                flapper.refresh_screen(stdscr)
                time.sleep(0.1)
    curses.wrapper(main)
class FlappyBird():
    high_score = 0
    def __init__(self):
        self.width = 20
        self.height = 20
        self.data = []
        self.bird_pos = round(self.height/2)
        self.velocity = -1
        self.score = 0
    def flap(self):
        self.velocity=2
    def make_pipe(self):
        hole = random.randint(0,self.width-5)+2
        self.data.append([hole,self.width-1])
    def refresh_screen(self, stdscr):
        disp_grid = []
        for i in range(self.width):
            disp_grid.append([])
            for j in range(self.height):
                disp_grid[i].append(0)
        for i in self.data:
            for j in range(len(disp_grid)):
                disp_grid[j][i[1]] = 1
            disp_grid[i[0]][i[1]] = 0
            disp_grid[i[0]+1][i[1]] = 0
            disp_grid[i[0]-1][i[1]] = 0
            disp_grid[i[0]+2][i[1]] = 0
            disp_grid[i[0]-2][i[1]] = 0
        try:
            disp_grid[self.bird_pos][5] = 2
        except:
            print()
        string = ""
        for i in range(len(disp_grid)):
            for j in range(len(disp_grid[i])):
                string+=str(disp_grid[i][j]).replace("0","--").replace("1","[]").replace("2","O>")
            string+="\n"
        string+=str(FlappyBird.high_score)
        string+="\n"+str(self.bird_pos)
        stdscr.clear()
        stdscr.addstr(1,0,string)
        stdscr.refresh()
    def update_objects(self):
        reward = 0
        if any((lst[1] == 5 for lst in self.data)):
            slist = next((lst for lst in self.data if lst[1] == 5))
            if (not (slist[0] >= self.bird_pos-2 and slist[0] <= self.bird_pos+2)) or self.bird_pos<=0 or self.bird_pos>=self.height:
                FlappyBird.high_score = max(self.score, FlappyBird.high_score)
                self.__init__()
                reward = -10
            else:
                reward = 10
                self.score+=1
        else:
            if self.bird_pos<=0 or self.bird_pos>=self.height:
                FlappyBird.high_score = max(self.score, FlappyBird.high_score)
                self.__init__()
                reward = -10
            else:
                reward = 1
        inputs = [self.bird_pos/self.height, [self.data[1][0] if self.data[0][1] <= 5 else self.data[0][0]]/self.height if self.data else 0, [self.data[1][1] if self.data[0][1] <= 5 else self.data[0][1]]/self.width if self.data else 0]
        self.bird_pos-=self.velocity
        if self.velocity > -2:
            self.velocity-=1
        remove = []
        for i in range(len(self.data)):
            if self.data[i][1] == 0:
                remove.append(self.data[i])
            else:
                self.data[i][1]-=1
        for i in remove:
            self.data.remove(i)
        return([reward,inputs])
