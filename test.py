import time
import os
import random
import tkinter as tk
import threading
import curses
def keyloop(ai):
    def update(c, flapper):
        result = ["", [0]]
        reward, inputs = flapper.update_objects()
        next_inputs = [flapper.bird_pos/flapper.height, (flapper.data[1][0] if len(flapper.data) >= 2 else flapper.data[0][0] if flapper.data[0][1] > 5 else 0)/flapper.height if flapper.data else 0, (flapper.data[1][1] if len(flapper.data) >= 2 else flapper.data[0][1] if flapper.data[0][1] > 5 else 0)/flapper.height if flapper.data else 0]
        if reward < 0:
            action = ai.choose_action(inputs)
            R = reward
            ai.punish(inputs, action, reward, next_inputs)
        elif next_inputs[2]*flapper.width-5 < abs(next_inputs[1]*flapper.height-next_inputs[0]*flapper.height)-2 and next_inputs[2]>0:
            action = ai.choose_action(inputs)
            R = -15
            ai.punish(inputs, action, -15, next_inputs)
        elif reward > 0:
            action = ai.choose_action(inputs)
            R = reward
            ai.reward(inputs, action, reward, next_inputs)
        if round(ai.choose_action(inputs)) >= 1:
            flapper.flap()
        if c%20==0:
            flapper.make_pipe()
        return R
    def main():
        with open('debug', 'w') as file:
            file.write("")
        #ai.load_weights()
        train_thread(0,1000000)
        #play()
    def train(cycles, thread_amt = 1, weight_file_name="weights"):
        threads = []
        chunk_size = cycles//thread_amt
        for i in range(thread_amt):
            start = i*chunk_size
            end = cycles if i==thread_amt-1 else (i+1) * chunk_size
            thread = threading.Thread(target=train_thread, args=(start,end))
            threads.append(thread)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        ai.save_weights(weight_file_name)
        print(f"Finished with a High Score of {FlappyBird.high_score}")
    def train_thread(start,end):
        flapper = FlappyBird()
        for i in range(start,end):
            update(i, flapper)
        ai.save_weights(weights)
    def play():
        flapper = FlappyBird()
        counter = 0
        while True:
            reward = update(counter)
            counter+=1
            flapper.refresh_screen(reward)
            time.sleep(0.1)

    main()
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
        self.velocity=1
    def make_pipe(self):
        hole = random.randint(0,self.width-5)+2
        self.data.append([hole,self.width-1])
    def refresh_screen(self, reward):
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
        string+="\n"+str(reward)
        os.system("clear")
        print(string)
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
        inputs = [self.bird_pos/self.height, (self.data[1][0] if self.data[0][1] <= 5 and len(self.data) >= 2 else self.data[0][0])/self.height if self.data else 0, (self.data[1][1] if self.data[0][1] <= 5 and len(self.data) >= 2 else self.data[0][1])/self.width if self.data else 0]
        self.bird_pos-=self.velocity
        if self.velocity == 1:
            self.velocity=-1
        remove = []
        for i in range(len(self.data)):
            if self.data[i][1] == 0:
                remove.append(self.data[i])
            else:
                self.data[i][1]-=1
        for i in remove:
            self.data.remove(i)
        return([reward, inputs])
