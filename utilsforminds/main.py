import utilsforminds
import numpy as np

# import plotly.plotly as py
from plotly.graph_objs import *
from plotly import tools as tls

import plotly.graph_objects as go

# x = np.arange(100)
# y = np.arange(100)
# z = np.arange(100)
# np.random.shuffle(x)
# np.random.shuffle(y)
# np.random.shuffle(z)

# points=Scatter3d(mode = 'markers',
#                  name = '',
#                  x =x,
#                  y= y,
#                  z= z,
#                  marker = Marker( size=2, color='#458B00' )
# )

# simplexes = Mesh3d(alphahull =1.0,
#                    name = '',
#                    x =x,
#                    y= y,
#                    z= z,
#                    color='blue', #set the color of simplexes in alpha shape
#                    opacity=0.15
# )

# x_style = dict( zeroline=False, range=[0, 200], tickvals=np.linspace(0, 200, 5)[1:].round(1))
# y_style = dict( zeroline=False, range=[0, 200], tickvals=np.linspace(0, 200, 4)[1:].round(1))
# z_style = dict( zeroline=False, range=[0, 200], tickvals=np.linspace(0, 200, 5).round(1))

# layout=Layout(title='Alpha shape of a set of 3D points. Alpha=0.1',
#               width=500,
#               height=500,
#               scene = Scene(
#               xaxis = x_style,
#               yaxis = y_style,
#               zaxis = z_style
#              )
# )

# fig=Figure(data=Data([points, simplexes]), layout=layout)
# fig.write_image("test_fig.pdf")
# fig.show()
