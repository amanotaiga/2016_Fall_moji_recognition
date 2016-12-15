import Tkinter as tk
from PIL import Image,ImageDraw
#from load_sample import get_testing_data
#from testing_sample_S import testing_result
from revised_load_data import get_testing_data
from revised_testing import testing_result
class MojiDrawing:
    def __init__(self,Groot):
        self.category = 2
        self.Groot = Groot
        self.canvas =tk.Canvas(self.Groot,width=64,height=64)
        self.canvas.place(x=60,y=50)
        self.canvas.bind("<B1-Motion>", self.motion)
        self.CheckVar1 =tk.IntVar()
        self.C1 = tk.Checkbutton(self.Groot, text = "Kata", variable = self.CheckVar1,
                 command = self.C1press,
                 width = 4)
        self.C1.place(x=100,y=170)
        self.CheckVar2 =tk.IntVar()
        self.C2 = tk.Checkbutton(self.Groot, text = "Hira", variable = self.CheckVar2,
                 command = self.C2press,
                 width = 4)
        self.C2.place(x=100,y=150)
        self.button = tk.Button(self.Groot,text="Save",width=5,bg='white',command=self.save)
        self.button.place(x=20,y=160)
        self.image=Image.new("RGB",(64,64))
        self.draw=ImageDraw.Draw(self.image)

    def save(self):
        filename = "Test_1.png"
        self.image.save(filename)
        self.Groot.destroy()
        get_testing_data()
        testing_result(self.category)
        if(self.category==0):
            print("Katakana") 
        elif(self.category==1):
            print("Hiragana")
        else:
            print("Please choose category")

    def C1press(self):
        self.category = 0
 
    def C2press(self):
        self.category = 1

    def motion(self,event):
        x1, y1 = ( event.x - 1 ), ( event.y - 1 )
        x2, y2 = ( event.x + 1 ), ( event.y + 1 )
        event.widget.create_oval( x1, y1, x2, y2,activefill="white",outline="white")
        self.draw.line(((x1,y1),(x2,y2)),(255,255,255),width=3)

if __name__ == "__main__":
    root=tk.Tk()
    root.wm_geometry("%dx%d+%d+%d" % (200, 200, 10, 10))
    root.config(bg='gray')
    MojiDrawing(root)
    root.mainloop()




