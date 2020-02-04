##Dash tutorial project

import dash
from dashboard.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()

app.layout = html.Div(children = [html.H1('Dash Tutorials'),
                                  dcc.Graph(id='example',
                                  figure={'data':[{'x':[1,2,3,4,1,2],'y':[3,4,5,8,6,8],'type':'line', 'name':'boats'},
                                                  {'x':[4,4,4,2,3,4],'y':[4.5,6,7,8,9],'type':'bar', 'name':'cars'},

                                                  ],
                                          'layout':{
                                            'title':'Basic Example'
                                          }

                                            })

])

if __name__ == '__main__':
    app.run_server(debug=True)
