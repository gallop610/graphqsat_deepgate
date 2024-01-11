import json
import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np

COLORS = ([
	# deepmind style
	'#0072B2',
	'#009E73',
	'#D55E00',
	'#CC79A7',
	'#F0E442',
	# built-in color
	'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
	'brown', 'orange', 'teal',  'lightblue', 'lime', 'lavender', 'turquoise',
	'darkgreen', 'tan', 'salmon', 'gold',  'darkred', 'darkblue',
	# personal color
	'#313695',  # DARK BLUE
	'#74add1',  # LIGHT BLUE
	'#4daf4a',  # GREEN
	'#f46d43',  # ORANGE
	'#d73027',  # RED
	'#984ea3',  # PURPLE
	'#f781bf',  # PINK
	'#ffc832',  # YELLOW
	'#000000',  # BLACK
])

def get_json(filePath):
    with open(filePath, "r", encoding="utf-8") as f:
        content = json.load(f)
    avgdata = {"step_ctr": [], "loss": [], "grad_norm": [], "lr": [], "average_q": []}
    maxdata = {"step_ctr": [], "loss": [], "grad_norm": [], "lr": [], "average_q": []}
    mindata = {"step_ctr": [], "loss": [], "grad_norm": [], "lr": [], "average_q": []}
    
    # 从50个元素的区间中取最大值/最小值/平均值
    for idx in range(len(content)):
        if(idx != 0 and idx % 50 == 0):
            el_list = content[idx-50:idx]
            avgdata['step_ctr'].append(np.average([el['step_ctr'] for el in el_list]))
            avgdata['loss'].append(np.average([el['loss'] for el in el_list]))
            avgdata['average_q'].append(np.average([el['average_q'] for el in el_list]))
            
            maxdata['step_ctr'].append(np.max([el['step_ctr'] for el in el_list]))
            maxdata['loss'].append(np.max([el['loss'] for el in el_list]))
            maxdata['average_q'].append(np.max([el['average_q'] for el in el_list]))
            
            mindata['step_ctr'].append(np.min([el['step_ctr'] for el in el_list]))
            mindata['loss'].append(np.min([el['loss'] for el in el_list]))
            mindata['average_q'].append(np.min([el['average_q'] for el in el_list]))
        # elif(idx >= 5000 and idx % 200 == 0):
        #     el_list = content[idx-200:idx]
        #     avgdata['step_ctr'].append(np.average([el['step_ctr'] for el in el_list]))
        #     avgdata['loss'].append(np.average([el['loss'] for el in el_list]))
        #     avgdata['average_q'].append(np.average([el['average_q'] for el in el_list]))
            
        #     maxdata['step_ctr'].append(np.max([el['step_ctr'] for el in el_list]))
        #     maxdata['loss'].append(np.max([el['loss'] for el in el_list]))
        #     maxdata['average_q'].append(np.max([el['average_q'] for el in el_list]))
            
        #     mindata['step_ctr'].append(np.min([el['step_ctr'] for el in el_list]))
        #     mindata['loss'].append(np.min([el['loss'] for el in el_list]))
        #     mindata['average_q'].append(np.min([el['average_q'] for el in el_list]))
    return avgdata, maxdata, mindata

def plot_reward():
    info_json = os.path.join(os.getcwd(), 'json/info.json')
    
    avgdata, maxdata, mindata = get_json(info_json)
    
    # plot step & loss
    plt.figure(figsize=(10, 6), dpi=600)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.plot(avgdata['step_ctr'], avgdata['loss'], color='#e75840', linestyle='-', label='Loss')
    plt.fill_between(avgdata['step_ctr'], maxdata['loss'], mindata['loss'], alpha=0.6, facecolor='#e75840')
    plt.savefig('figure/loss.png', dpi=600)
    plt.close()
    
    # plot step & average Q_value
    plt.xlabel("Step")
    plt.ylabel("Average Q-Value")
    plt.plot(avgdata['step_ctr'], avgdata['average_q'], color='#628cee', linestyle='-', label='Average Q')
    plt.fill_between(avgdata['step_ctr'], maxdata['average_q'], mindata['average_q'], alpha=0.6, facecolor='#628cee')
    plt.savefig('figure/average_q.png', dpi=600)

if __name__ == "__main__":
	plot_reward()