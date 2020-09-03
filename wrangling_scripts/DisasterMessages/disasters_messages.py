from wrangling_scripts.DisasterMessages.train_classifier import *
from plotly.graph_objs import Bar, Heatmap

def return_figures_disaster():
    # load data
    engine = create_engine(
        'sqlite:///data/DisasterMessages/DisasterResponse.db')

    df_data = pd.read_sql("SELECT * FROM DisasterMessages", engine)
    df_word = pd.read_sql("SELECT * FROM Words", engine)
    category_names = list(df_data.columns[4:])
        
    #update custom pickler
    class CustomUnpickler(pickle.Unpickler):

        def find_class(self, module, name):
            if name == 'tokenize':
                from wrangling_scripts.DisasterMessages.train_classifier import tokenize, get_wordnet_pos
                return tokenize
            return super().find_class(module, name)

    model = CustomUnpickler(open('wrangling_scripts/DisasterMessages/model/model_ada.sav', 'rb')).load()

    # selecting categories for visualisation
    df_word_subset = df_word[df_word['category_name'].isin(['cold', 'storm', 'shelter', 'weather_related', 'clothing', 'infrastructure_related', 'buildings'])]

    # get unique words with high training weight, and their categories
    unique_category = list(set(df_word_subset.category_name))
    unique_word = list(set(df_word_subset.important_word))

    # word dictionary with categories one-hot coded
    dict_category_word = {}
    for category in unique_category:
        dict_category_word[category] = []
    for category in unique_category:
        sub_sub = df_word_subset[df_word_subset.category_name == str(category)]
        for word in unique_word:
            if(word in list(sub_sub.important_word.values)):
                dict_category_word[category].append(np.round(
                    sub_sub[sub_sub.important_word == str(word)].importance_value.values[0], 2))
            else:
                dict_category_word[category].append(float(0))

    category_active = [np.round(df_data[str(category)].sum(
    ) * 100 / df_data.shape[0]) for category in category_names]


    # get arrays for heat map plotting
    heatmap_array = []
    for category in unique_category:
        heatmap_array.append(dict_category_word[category])

    genre_counts = df_data.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)


    # Graph1
    graph_one = []
    graph_one.append(
            Bar(
                x=genre_names,
                y=genre_counts, marker=dict(
                color='rgba(222,45,38,0.8)'), opacity=0.6,
            )
        )

    layout_one = dict(title='Distribution of Message Genres',
                      xaxis=dict(title='Genre'),
                      yaxis=dict(title='Count'),
                      font=dict(size=18),
                      )

        # Graph2
    graph_two = []
    graph_two.append(
            Bar(
                x=category_names,
                y=category_active, marker=dict(
                color='rgba(222,45,38,0.8)'), opacity=0.6,
            )
        )

    layout_two = dict(title='Percent of True sample over all sample, per Message category',
                      font=dict(size=15),
                      xaxis=dict(tickangle=-30, automargins=True, font=dict(size=6),autotick=False),  #autotick=False, , 
                      yaxis=dict(title='% of True samples',
                      automargins=True), 

                      )

    # Graph3
    graph_three = []
    if 'cold' in unique_category:
        trace1 = Bar(
            x=unique_word,
            y=dict_category_word['cold'],
            marker=dict(color='rgba(222,45,38,0.8)'),
            opacity=0.6,
            name='Cold',
            text='Cold',
            width=0.3
            # orientation = 'h'
        )
        graph_three.append(trace1)

    if 'storm' in unique_category:
        trace2 = Bar(
            x=unique_word,
            y=dict_category_word['storm'],
            marker=dict(color='rgb(49,130,189)'),
            opacity=0.76,
            name='Storm',
            text='Storm',
            width=0.3
            # orientation = 'h'
        )
        graph_three.append(trace2)
        
    if 'shelter' in unique_category:
        trace3 = Bar(
            x=unique_word,
            y=dict_category_word['shelter'],
            marker=dict(color='rgb(204,204,204)'),
            opacity=0.9,
            name='Shelter',
            text='Shelter',
            width=0.3
            # orientation = 'h'
        )
        graph_three.append(trace3)

    if 'weather_related' in unique_category:
        trace4 = Bar(
            x=unique_word,
            y=dict_category_word['weather_related'],
            marker=dict(color='rgb(244,109,67)'),
            opacity=0.4,
            name='Weather_related',
            text='Weather_related',
            width=0.3
            # orientation = 'h'
        )
        graph_three.append(trace4)

    if 'clothing' in unique_category:
        trace5 = Bar(
            x=unique_word,
            y=dict_category_word['clothing'],
            marker=dict(color='rgb(102,205,170)'),
            opacity=0.6,
            name='Clothing',
            text='Clothing',
            width=0.3
            # orientation = 'h'
        )
        graph_three.append(trace5)
        
    if 'infrastructure_related' in unique_category:
        trace6 = Bar(
            x=unique_word,
            y=dict_category_word['infrastructure_related'],
            marker=dict(color='rgb(100,149,237)'),
            opacity=0.6,
            name='Infrastructure_related',
            text='Infrastructure_related',
            width=0.3
            # orientation = 'h'
        )
        graph_three.append(trace6)
        
    if 'buildings' in unique_category:
        trace7 = Bar(
            x=unique_word,
            y=dict_category_word['buildings'],
            marker=dict(color='rgb(160,82,45)'),
            opacity=0.6,
            name='Buildings',
            text='Buildings',
            width=0.3
            # orientation = 'h'
        )
        graph_three.append(trace7)

    layout_three = dict(title='Words importances per category after training (few columns)',
                        xaxis=dict(autotick=False, tickangle=-35,),
                        yaxis=dict(title='Weights', automargins=True),
                        hovermode='closest',
                        font=dict(size=18),  # barmode='group',
                        )


    # Graph4
    graph_four = [Heatmap(z=heatmap_array,
                          x=unique_word,
                          y=unique_category,
                          opacity=0.6,
                          xgap=3,
                          ygap=3,
                          colorscale='Jet')]

    layout_four = dict(
        title='Few category name vs. their most important words after training',
        xaxis=dict(showline=False, showgrid=False, zeroline=False,),
        yaxis=dict(showline=False, showgrid=False, zeroline=False),
        font=dict(size=18),
        plot_bgcolor=('#fff'), height=500,
        
    )

    # add plots/layouts in arrays for Json dump
    graphs = []
    graphs.append(dict(data=graph_four, layout=layout_four))
    graphs.append(dict(data=graph_two, layout=layout_two))
    graphs.append(dict(data=graph_three, layout=layout_three))
    graphs.append(dict(data=graph_one, layout=layout_one))


    return graphs, model, df_data