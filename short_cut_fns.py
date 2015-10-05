import matplotlib.pyplot as plt
import numpy as np
import galsim
def simple_sublot(data,names,indices,titles=['']*5,xlabs=['']*5,ylabs=['']*5,legends=['']*5,colors=['r','b','g','c','y'],xlims=[False]*5,ylims=[False]*5,figsize=[12,10],space=[0.4,0.6],n_columns=2):
    '''Defalut plot parametrs are for 5 subplots and 5variables per plot. Function must me provided with color and legends for more 
        Data must be of the form data{x: {index1: [common xaxis subplot1];index2=[common xaxis subplot2]....};
                                      y1:{index1:[y1 subplot1];           index2=[y1 subplot2];........      };
                                      y2:{index1:[y2 subplot1];           index2=[y2 subplot2];........      };
        names=[x,y1,y2...]; Number of plots per subplot(len(names)-1)
              Note:names[0] must be the xaxis
        index=[index1,index2..]; Number of sublots is len(indices)
        titles:    title for each subplot
        xlabs:     array containg x-axis label for each sublot(len(xlabs)=len(indices))
        xlabs:     array containg y-axis label for each sublot(len(ylabs)=len(indices))
        legends:   label for each plot per subplot(len(legends)=len(names)-1)
        colors:    Color for each plot per subplot(len(legends)=len(names)-1)
        xlims:     array of limits on x-axis per subplot
        ylims:     array of limits on y-axis per subplot
        figsize:   Size of figure; default:[12,10]
        space:     [wspace ,hspace] for subplots; default:[0.4,0.6]
        n_columns: Number of columns in subplot;default=2
        Exampe:
        x,a,b,c={},{},{},{}
        x['1'],x['2'],x['3']=range(5),range(10),np.linspace(0,360,100)*np.pi/180.
        a['1'],a['2'],a['3']=range(5),range(10),np.cos(x['3'])
        b['1'],b['2'],b['3']=np.array(range(5))*5,np.array(range(10))*3,np.sin(x['3'])
        c['x']=x
        c['a']=a
        c['b']=b
        simple_sublot(c,['x','a','b'],['1','2','3'],titles=['plot1','plot2','plot3'],n_columns=3,legends=['a','b'],figsize=[14,12],ylims=[[0,10],[0,10],[-1,1]],xlims=[[0,4],[0,4],[0,2*np.pi]])
        '''
    plt.figure(1,figsize=figsize)
    plt.subplots_adjust(wspace = space[0])
    plt.subplots_adjust(hspace = space[1])
    n_plots=len(indices)
    for i in range(0,n_plots):
        plt.subplot(n_plots,n_columns,i+1)
        n_s=0
        for name in names[1:]:
            plt.plot(data[names[0]][indices[i]],data[name][indices[i]],colors[n_s],label=legends[n_s])   
            n_s+=1
        if (xlims[i] != False):
            plt.xlim(xlims[i])
        if (ylims[i] != False):
            plt.ylim(ylims[i])
        plt.ylabel(ylabs[i])
        plt.xlabel(xlabs[i])
        plt.title(titles[i])
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
class simple_image:
    def __init__(self,
                 I_size=[128,128], 
                 pixel_scale=None,
                 method='auto',
                 T_fn=None
                ):
        """Class to store parametrs used to draw images
        I_size     : Size of image; default=[128,128]
        pixel_scale: Scale to draw the image, in arcsec/pixel;default lets galsim pick scale depending on Nyquist scale
        method: """
        self.size=I_size
        self.scale=pixel_scale
        self.method=method
        self.filt=T_fn
    def draw_image(self):
        return galsim.ImageF(self.size[0],self.size[1],scale=self.scale)

