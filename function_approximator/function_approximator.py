
import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import numpy as np

import inspect
import threading
import queue
import matplotlib.cm as cm
import itertools

torch.manual_seed(1)
plt.style.use('dark_background')
sg.theme('Black')

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


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if (v.default is not inspect.Parameter.empty) and (k != 'lr')
    }


x, y = '',''
new_x,new_y = '',''
epoch_num = ''

def train_network(net_fig_queue,stop_training_queue):
    
    data_loader = DataLoader(data ,data_points)
        
    neurons = int(values['neuron_slider'] )
    hidden_layers = int(values['hidden_slider'])
    learning_rate = float(values['lr_slider']) 
    epochs = int(values['epoch_slider'])

    model = my_model(num_layers = hidden_layers, width = neurons)
    criterion = nn.MSELoss()

    parameter_tuple = tuple(optimizer_parameter_dict[values['optimizer']].values())
    optimizer = getattr(torch.optim,values['optimizer'])(model.parameters(),learning_rate,*parameter_tuple)
    

    for i in range(epochs):
        for x,y in data_loader:
            y_pred = model.forward(x)
            loss = criterion(y_pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item()) 
 

        try:
            train_terminate = stop_training_queue.get_nowait()
        except queue.Empty:
            train_terminate = None
        finally:
            if train_terminate:
                break
            elif i%int(values['plot_step']) == 0:
                global new_x,new_y,epoch_num
                epoch_num = i
                new_x, new_y = zip(*sorted(zip(x.numpy(), y_pred.data.numpy())))
                net_fig_queue.put(1)
            

    window['stop_training'].update(disabled = True)
    


text_len= len('GOOEY Neural Nets: Function Approximators')
data_frame_text_len = len('Number of clusters:')
model_frame_text_len = len('Number of neurons in layer:')
optimizer_list = [optimizer for optimizer in  dir(torch.optim) if (optimizer not in ['Optimizer','lr_scheduler']) and ('__' not in optimizer  ) ]
max_optim_len = max([len(string) for string in optimizer_list]) + 1
optimizer_parameter_dict = {optimizer:get_default_args(getattr(torch.optim,optimizer)) for optimizer in optimizer_list }
initial_optim_params_list = list( optimizer_parameter_dict[optimizer_list[0]] )
init_params_dict = optimizer_parameter_dict[optimizer_list[0]]
max_optim_len = max([len(string) for string in optimizer_list])




layout = [

    [sg.Text('GOOEY Neural Nets: Function Approximators', size=(text_len, 1), justification='center', font=("Arial", 25), relief=sg.RELIEF_RIDGE)],
    
    [sg.Frame(layout = [
    
        [sg.Text('Datapoints:',size = (data_frame_text_len,1)), sg.Slider(default_value = 1000,range=(0, 100000), size=(60, 20), orientation='h', key='data_slider')],
        [sg.Text('noise in data:',size = (data_frame_text_len,1)), sg.Slider(default_value = 0.5,range=(0, 100) , resolution = 0.1, size=(60, 20), orientation='h', key='noise_slider')],
        [sg.Text('function order:',size = (data_frame_text_len,1)) , sg.DropDown(values = list(range(1,11)) , default_value = 2, size = (5,1),key='func_order' ) ]
        
    
    ], relief = sg.RELIEF_SUNKEN , title = 'DATA'  )   ],
    
    [sg.Frame(layout = [
    
        [sg.Text('Number of neurons in layer:',size = (model_frame_text_len,1) ),sg.Slider(default_value = 2, range=(1, 100), size=(60, 20), orientation='h', key='neuron_slider')     ],
        [sg.Text('Number of hidden layers:',size = (model_frame_text_len,1)),sg.Slider(default_value = 2,range=(1, 100), size=(60, 20), orientation='h', key='hidden_slider')     ],
        [sg.Text('Learning Rate:',size = (model_frame_text_len,1)) ,sg.Slider(default_value = 0.01,range=(0, 10),resolution=0.01, size=(60, 20), orientation='h', key='lr_slider')    ],
        [sg.Text('Epochs:',size = (model_frame_text_len,1)),sg.Slider(default_value = 1000,range=(0, 10000),resolution=1, size=(60, 20), orientation='h', key='epoch_slider')     ],
        [sg.Text('Optimizers:',size = (model_frame_text_len,1)),sg.DropDown(values = optimizer_list , default_value = optimizer_list[0],enable_events = True, key = 'optimizer',size = (max_optim_len,1))],
        [sg.Text('Optimizer parameters:',size = (model_frame_text_len,1)), sg.DropDown(values = initial_optim_params_list, default_value = initial_optim_params_list[0], enable_events = True  , key ='optimizer_params')],
        [sg.Text('parameter value:',size = (model_frame_text_len,1)),sg.Input( init_params_dict[ initial_optim_params_list[0]],tooltip = 'Enter in format shown', key = 'param_value',size = (10,1)) , sg.Text(type(init_params_dict[ initial_optim_params_list[0]]),key='param_type' ) , sg.Button('Update',key = 'update') ,sg.Button('reset all',key = 'reset')]
        
        
    
    ], relief = sg.RELIEF_SUNKEN , title = 'MODEL'  )   ],
    
    [sg.Button('Generate Data',key = 'gen_data'),sg.Button('Start Training',disabled=True,key = 'train_net'), sg.Text('Plot step size:') , sg.DropDown(values = list(range(0,1000,10)) ,default_value= 10,key = 'plot_step' ) ,sg.Button('Stop Training' , disabled = True, key = 'stop_training') ],
    [sg.Canvas( size = (100,100) , key = 'net_canvas' ),sg.Canvas( size = (100,100) , key = 'loss_canvas' )],

    
    [sg.Button('Exit')]
]      

window = sg.Window('GOOEY Neural Nets', layout , finalize = True) 

# net canvas
net_canvas_elem = window['net_canvas']
net_canvas = net_canvas_elem.TKCanvas

net_fig = plt.figure(figsize=(4,4))
net_plot = net_fig.add_subplot(111)
# net_plot.axis('off')
net_fig_agg = draw_figure(net_canvas, net_fig)

# loss canvas
loss_canvas_elem = window['loss_canvas']
loss_canvas = loss_canvas_elem.TKCanvas

loss_fig = plt.figure(figsize=(4,4))
loss_plot = loss_fig.add_subplot(111)
# loss_plot.axis('off')
loss_fig_agg = draw_figure(loss_canvas, loss_fig)

# plt.tight_layout()

draw_signal = queue.Queue()
stop_training = queue.Queue()

# =======
model = ''
criterion = ''
optimizer = ''
epochs = ''

losses = []

data_points = 0
data = ''

while 1:
    event, values = window.read(timeout = 100)    
    if event == 'Exit':
        stop_training.put(1)
        
        try:
            t1.join()
        except:
            pass
        window.close()
        break
    elif event == 'optimizer':
        
         window['optimizer_params'].update( values= list( optimizer_parameter_dict[ values['optimizer'] ] ) ) 

    elif event == 'optimizer_params':
        
        current_param_type = type(optimizer_parameter_dict[ values['optimizer'] ][values['optimizer_params']])
        window['param_type'].update(current_param_type if current_param_type != int else float)
        param_value = optimizer_parameter_dict[ values['optimizer'] ][values['optimizer_params']]
        window['param_value'].update( param_value if current_param_type != int else float(param_value) )
       
        
    elif event == 'update':
        current_param_type = type( optimizer_parameter_dict[ values['optimizer'] ][values['optimizer_params']] )
        new_value = ''
        if current_param_type == tuple:
            indiv_type = type(optimizer_parameter_dict[ values['optimizer'] ][values['optimizer_params']][0])
            if len(values['param_value'].split()) != 2:
                print('incorrect input')
            else:
                new_value = ( indiv_type(values['param_value'].split()[0]) , indiv_type(values['param_value'].split()[1]) )
                optimizer_parameter_dict[ values['optimizer'] ][values['optimizer_params']] = new_value
        else:
            try:
                new_value = current_param_type(values['param_value'])
                optimizer_parameter_dict[ values['optimizer'] ][values['optimizer_params']] = new_value
            except:
                print('incorrect input')
                
    elif event == 'gen_data':
        net_plot.cla()                   
        new_x ,new_y = '',''
#         net_plot.axis('off')
        
        data_points = int(values['data_slider']) # draw this many data points (on next line)
        noise =  float(values['noise_slider'])
        order = int(values['func_order'])
        
        data = Dataset(num_data_points = data_points, std_dev = noise , order = order)
        
        x,y = data.get_all_data()
        
        net_plot.scatter(x, y, color='yellow')
        
        net_fig_agg.draw()
        window['train_net'].update(disabled = False)
        
    elif event == 'train_net':
        losses = []
        t1 = threading.Thread(target=train_network , args = (draw_signal,stop_training) , daemon = True)
        t1.start()
        window['stop_training'].update(disabled = False)
    elif event =='stop_training':  
        stop_training.put(1)
        window['stop_training'].update(disabled = True)
        
        
    try:
        draw_status = draw_signal.get_nowait()
    except queue.Empty:
        draw_status = None

    if draw_status:

        net_plot.cla()
        net_plot.set_title(f'epoch:{epoch_num}')
        net_plot.plot(x.numpy(),y.numpy(),'o', color = 'yellow')

        net_plot.plot(new_x ,new_y,color = 'green',lw = '3')
        net_fig_agg.flush_events()
        net_fig_agg.draw()

        loss_plot.cla()
        loss_plot.set_yticklabels([])
        loss_plot.set_title(f'loss:{round(losses[-1],4)}')
        loss_plot.plot(np.array(losses))
        loss_fig_agg.flush_events()
        loss_fig_agg.draw()

        


