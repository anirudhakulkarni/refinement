import numpy as np
import os
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd


# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')

app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='graph'),
    dcc.Checklist(
        id="checklist",
        options=["MMCE", "FL+MDCA", "FLSD", "LS", "LS+MDCA", "brierscore",
                 "NLL+MDCA", "NLL+DCA", "crossentropy", "focalloss"],
        value=["brierscore"],
        inline=True
    ),
    dcc.RadioItems(
        id="radio",
        options=["svhn", "cifar10", "cifar100"],
        value="cifar10",
        labelStyle={'display': 'inline-block'}
    ),

])

def get_trained_loss(checkpoint):
    trained_loss = ("".join(checkpoint.split('.')[0].split('_')[2:]))[:-12]
    return trained_loss


root = "./aurocs/"
plots = {}
for file in os.listdir(root):
    if not ".png" in file and os.path.isfile(root+file):
        data = np.load(root+file)
        # file= resnet56_svhn_NLL+MDCA_25-May_tpr_fpr.npy, trained_loss=NLL+MDCA, model=resnet56, dataset=svhn
        trained_loss = get_trained_loss(file)
        model = file.split('_')[0]
        dataset = file.split('_')[1]
        if model+'_'+dataset not in plots:
            plots[model+'_'+dataset] = {}
        plots[model+'_'+dataset][trained_loss] = {"x":list(data[0:len(data)//2]),"y":list(data[len(data)//2:])}
# print(plots)
@app.callback(
    Output('graph', 'figure'),
    [Input('checklist', 'value'),
     Input('radio', 'value')])
def update_figure(checklist, radio):
    # filtered_df = df[df.year == selected_year]
    data = plots["resnet56_"+radio]
    column_names=["loss","y","x"]
    df = pd.DataFrame(columns=column_names)
    # print(checklist)
    for d in data:
        # print(data[d])
        if d in checklist:
            # print(0)
            # insert row in df
            loss_column = [d for i in range(len(data[d]["x"]))]
            # create y_column with all rows=d
            data[d]["loss"] = loss_column
            d2=pd.DataFrame(columns=column_names,data=data[d])
            df=pd.concat([df,d2])
            # df.insert(0,data[d])
            # df.insert(0, d, data[d]["x"])
            # df = df.insert(data[d])
    # print(df)
    fig = px.scatter(df, x="x", y="y", color="loss",title="auroc_"+radio+"_resenet56")

    fig.update_layout(transition_duration=500)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
