import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)



class Dataset(BaseDataset):

    def __init__(self, num_data_points = 100, std_dev = 1 , order = 1 ):
        
        self.x = torch.randn(num_data_points,1)
        self.y = 0
        for i in range(1,order+1):
            self.y += self.x**i 
        self.y += std_dev*torch.randn(num_data_points,1)
        
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
    def get_all_data(self):
        return self.x,self.y
    
    def visualize(self):
        plt.plot(self.x.numpy(),self.y.numpy() , 'o')
        plt.show()
    
    def __len__(self):
        return len(self.y)

    
class my_model(nn.Module):
    def __init__(self, input_size = 1 , num_layers = 1, width = 1):
        super(my_model,self).__init__()
        self.input_layer = nn.Linear(input_size,width)
        self.hidden_layer = nn.Linear(width,width)
        self.output_layer = nn.Linear(width,1)
        self.num_layers = num_layers
    def forward(self,x):
        x = F.relu(self.input_layer(x))
        for i in range(self.num_layers):
            x = F.relu(self.hidden_layer(x))
        pred = self.output_layer(x)
        return pred
    
    

def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


text_len= len('GOOEY Neural Nets: Function Approximators')
data_frame_text_len = len('function order:')
model_frame_text_len = len('Number of neurons in layer:')



layout = [

    [sg.Text('GOOEY Neural Nets: Function Approximators', size=(text_len, 1), justification='center', font=("Helvetica", 25), relief=sg.RELIEF_RIDGE)],
    
    [sg.Frame(layout = [
    
        [sg.Text('Datapoints:',size = (data_frame_text_len,1)), sg.Slider(default_value = 1000,range=(0, 100000), size=(60, 20), orientation='h', key='data_slider')],
        [sg.Text('noise in data:',size = (data_frame_text_len,1)), sg.Slider(default_value = 0.5,range=(0, 100) , resolution = 0.1, size=(60, 20), orientation='h', key='noise_slider')],
        [sg.Text('function order:',size = (data_frame_text_len,1)) , sg.DropDown(values = list(range(1,11)) , default_value = 2, size = (5,1),key='func_order' ) ]
        
    
    ], relief = sg.RELIEF_SUNKEN , title = 'DATA'  )   ],
    
    [sg.Frame(layout = [
    
        [sg.Text('Number of neurons in layer:',size = (model_frame_text_len,1) ),sg.Slider(default_value = 2, range=(0, 1000), size=(60, 20), orientation='h', key='neuron_slider')     ],
        [sg.Text('Number of hidden layers:',size = (model_frame_text_len,1)),sg.Slider(default_value = 2,range=(0, 1000), size=(60, 20), orientation='h', key='hidden_slider')     ],
        [sg.Text('Learning Rate:',size = (model_frame_text_len,1)) ,sg.Slider(default_value = 0.01,range=(0, 10),resolution=0.01, size=(60, 20), orientation='h', key='lr_slider')    ],
        [sg.Text('Epochs:',size = (model_frame_text_len,1)),sg.Slider(default_value = 1000,range=(0, 100000),resolution=1, size=(60, 20), orientation='h', key='epoch_slider')     ],
        
    
    ], relief = sg.RELIEF_SUNKEN , title = 'MODEL'  )   ],
    
    [sg.Button('Generate Data',key = 'gen_data'),sg.Button('Start Training',disabled=True,key = 'train_net')],
    [sg.Canvas( size = (100,100) , key = 'data_canvas' ) ,sg.Canvas( size = (100,100) , key = 'net_canvas' ) ],

    
    [sg.Button('Exit')]
]      
sg.theme('DarkBlue')
window = sg.Window('GOOEY Neural Nets', layout , finalize = True) 

data_canvas_elem = window['data_canvas']
data_canvas = data_canvas_elem.TKCanvas

data_fig = plt.figure(figsize=(4,4))
data_plot = data_fig.add_subplot(111)
data_plot.grid()

data_fig_agg = draw_figure(data_canvas, data_fig)



net_canvas_elem = window['net_canvas']
net_canvas = net_canvas_elem.TKCanvas

net_fig = plt.figure(figsize=(4,4))
net_plot = net_fig.add_subplot(211)
loss_plot = net_fig.add_subplot(212)
net_plot.grid()
plt.tight_layout()

net_fig_agg = draw_figure(net_canvas, net_fig)


model = my_model()
criterion = nn.MSELoss()
optimizer = torch.optim.Adagrad(model.parameters() , lr = 0.01)
epochs = 1000
losses = []

data_points = 0
data = ''
while 1:
    event, values = window.read()    
    if event == 'Exit':
        break
    elif event == 'gen_data':
        data_plot.cla()                    # clear the subplot
        data_plot.grid()                   # draw the grid
        data_points = int(values['data_slider']) # draw this many data points (on next line)
        noise =  float(values['noise_slider'])
        order = int(values['func_order'])
        data = Dataset(num_data_points = data_points, std_dev = noise , order = order)
        x,y = data.get_all_data()
        data_plot.scatter(x, y, color='purple')
        data_fig_agg.draw()
        window['train_net'].update(disabled = False)
        
    elif event == 'train_net':
        
        data_loader = DataLoader(data ,data_points)
        
        neurons = int(values['neuron_slider'] )
        hidden_layers = int(values['hidden_slider'])
        learning_rate = float(values['lr_slider']) 
        epochs = int(values['epoch_slider'])
        
        model = my_model(num_layers = hidden_layers, width = neurons)
        criterion = nn.MSELoss()

        optimizer = torch.optim.Adagrad(model.parameters() , lr = learning_rate)
        losses = []
        
        for i in range(epochs):
            for x,y in data_loader:
                y_pred = model.forward(x)
                loss = criterion(y_pred,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss) 
            #     if i%100 == 0:
            #         print(f'epoch:{i} loss:{loss.item()}')

                if i%10 == 0:
                    net_plot.cla()
                    net_plot.set_title(f'epoch:{i}')
                    net_plot.plot(x.numpy(),y.numpy(),'o')
                    net_plot.plot(x.numpy(),y_pred.data.numpy(),'go')
                    loss_plot.cla()
                    loss_plot.set_title(f'loss:{round(loss.item(),4)}')
                    loss_plot.plot(np.array(losses)/len(losses) )
                    net_fig_agg.draw()
#                     plt.pause(0.001)

        
window.close()

