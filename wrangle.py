def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import re
import time
import numpy as np
import pandas as pd
from itertools import combinations
import nltk
nltk.download("vader_lexicon")
from nltk.sentiment import SentimentIntensityAnalyzer
import scipy.cluster.hierarchy as sch
import plotly.express as px
from plotly.offline import plot

if os.name == "nt":
    path_sep = "\\"
else:
    path_sep = "/"


def prepare(df, name="Data Preparation", path=None, plots=True):
    data = Prepare(df, name, path, plots)
    print("Data Wrangling:")
    start = time.time()
    data.drop()
    data.product_name()
    data.category()
    data.actual_price()
    data.discount_percentage()
    data.rating()
    data.rating_count()
    data.about_product_positivity()
    data.review_title_positivity()
    data.review_content_positivity()
    data.aggregate()
    data.shuffle()
    end = time.time()
    data.run_time(start, end)
    
    if plots:
        print("Plotting:")
        start = time.time()
        data.separate()
        data.correlations()
        data.scatter_plots()
        data.bar_plots()
        data.pairwise_bar_plots()
        data.boxplots()
        end = time.time()
        data.run_time(start, end)

    return data.df
    
    
class Prepare:
    def __init__(
        self, 
        df,
        name="Data Preparation", 
        path=None,
        plots=True,
    ):
        self.df = df  # dataset
        self.name = name  # name of the analysis
        self.path = path  # the path where results will be exported
        self.plots = plots  # should we plot the analysis?
        
        if self.path is None:
            self.path = os.getcwd()

        # create folders for output files
        if self.plots:
            self.folder(f"{self.path}{path_sep}{self.name}")
    
    def drop(self):
        print("> Removing Unnecessary Columns")
        self.df = self.df.drop(columns=[
            "discounted_price", 
            "user_id", 
            "user_name", 
            "review_id", 
            "img_link", 
            "product_link",
        ])
    
    def product_name(self):
        print("> Transforming Product Name")
        # get the first word from product_name
        word = list()
        for item in self.df["product_name"]:
            word.append(item.split(" ")[0])
        self.df["product_name"] = word

    def category(self):
        print("> Transforming Category")
        # get the first section from category
        section = list()
        for item in self.df["category"]:
            section.append(item.split("|")[0])
        self.df["category"] = section

    def actual_price(self):
        print("> Transforming Actual Price")
        # convert actual_price to a float
        self.df["actual_price"] = self.df["actual_price"].str.replace("₹", "")
        self.df["actual_price"] = self.df["actual_price"].str.replace(",", "")
        self.df["actual_price"] = self.df["actual_price"].astype(float)

    def discount_percentage(self):
        print("> Transforming Discount Percentage")
        # convert discount_percentage to a float
        self.df["discount_percentage"] = self.df["discount_percentage"].str.replace("%", "")
        self.df["discount_percentage"] = self.df["discount_percentage"].astype(float) / 100
        
    def rating(self):
        print("> Transforming Rating")
        # convert rating to a float
        self.df["rating"] = self.df["rating"].str.replace("|", "NaN")
        self.df["rating"] = self.df["rating"].astype(float)

    def rating_count(self):
        print("> Transforming Rating Count")
        # convert rating_count to a float
        self.df["rating_count"] = self.df["rating_count"].str.replace(",", "")
        self.df["rating_count"] = self.df["rating_count"].astype(float)

    def about_product_positivity(self):
        print("> Computing About Product Positivity")
        self.sia = SentimentIntensityAnalyzer()
        positivity = list()
        for item in self.df["about_product"]:
            positivity.append(self.sia.polarity_scores(item)["compound"])
        self.df["about_product_positivity"] = positivity
        self.df = self.df.drop(columns="about_product")
    
    def review_title_positivity(self):
        print("> Computing Review Title Positivity")
        positivity = list()
        for item in self.df["review_title"]:
            positivity.append(self.sia.polarity_scores(item)["compound"])
        self.df["review_title_positivity"] = positivity
        self.df = self.df.drop(columns="review_title")

    def review_content_positivity(self):
        print("> Computing Review Content Positivity")
        positivity = list()
        for item in self.df["review_content"]:
            positivity.append(self.sia.polarity_scores(item)["compound"])
        self.df["review_content_positivity"] = positivity
        self.df = self.df.drop(columns="review_content")
    
    def aggregate(self):
        print("> Aggregating By Product ID")
        # aggregate the strings by product_id
        strings = self.df.groupby("product_id")[["product_id", "product_name", "category"]].head(1)
        strings = strings.sort_values(by="product_id").reset_index(drop=True)

        # aggregate the numbers by product_id
        self.df = self.df.groupby("product_id").agg({
            "actual_price": "mean",
            "discount_percentage": "mean",
            "rating": "mean",
            "rating_count": "mean",
            "about_product_positivity": "mean",
            "review_title_positivity": "mean",
            "review_content_positivity": "mean",
        }).reset_index()
        self.df = self.df.sort_values(by="product_id").reset_index(drop=True)

        # add the strings back to the data
        self.df = pd.concat([strings, self.df.drop(columns="product_id")], axis="columns")

        # remove the product_id
        self.df = self.df.drop(columns="product_id")
    
    def shuffle(self):
        print("> Shuffling The Data")
        self.df = self.df.sample(frac=1, random_state=0).reset_index(drop=True)

    def separate(self):
        self.numbers = self.df.drop(columns=["product_name", "category"]).columns.tolist()
        self.strings = ["product_name", "category"]
        
    def correlations(self):
        if self.plots:
            print("> Plotting Correlations")
            self.correlation_plot(
                df=self.df[self.numbers], 
                title="Correlation Heatmap",
                font_size=16,
            )

    def scatter_plots(self):
        if self.plots:
            pairs = list(combinations(self.numbers, 2))
            for pair in pairs:
                print(f"> {pair[0]} vs. {pair[1]}")
                self.scatter_plot(
                    df=self.df,
                    x=pair[0],
                    y=pair[1],
                    color=None,
                    title=f"{pair[0]} vs. {pair[1]}",
                    font_size=16,
                )

    def histograms(self):
        if self.plots:
            for col in self.numbers:
                print(f"> Plotting {col}")
                self.histogram(
                    df=self.df,
                    x=col,
                    bins=20,
                    title=col,
                    font_size=16,
                )
                
    def bar_plots(self):
        if self.plots:
            for col in self.strings:
                print(f"> Plotting {col}")
                proportion = self.df[col].value_counts(normalize=True).reset_index()
                proportion.columns = ["Label", "Proportion"]
                proportion = proportion.sort_values(by="Proportion", ascending=False).reset_index(drop=True)
                self.bar_plot(
                    df=proportion,
                    x="Proportion",
                    y="Label",
                    title=col,
                    font_size=16,
                )

    def pairwise_bar_plots(self):
        if self.plots:
            pairs = list(combinations(self.strings, 2))
            for pair in pairs:
                print(f"> {pair[0]} vs. {pair[1]}")
                data = pd.DataFrame()
                data[f"{pair[0]}, {pair[1]}"] = self.df[pair[0]].astype(str) + ", " + self.df[pair[1]].astype(str)
                proportion = data[f"{pair[0]}, {pair[1]}"].value_counts(normalize=True).reset_index()
                proportion.columns = [f"{pair[0]}, {pair[1]}", "Proportion"]
                proportion = proportion.sort_values(by="Proportion", ascending=False).reset_index(drop=True)
                self.bar_plot(
                    df=proportion,
                    x="Proportion",
                    y=f"{pair[0]}, {pair[1]}",
                    title=f"{pair[0]} vs. {pair[1]}",
                    font_size=16,
                )

    def boxplots(self):
        if self.plots:
            pairs = list()
            for number in self.numbers:
                for string in self.strings:
                    pairs.append((number, string))
            
            for pair in pairs:
                print(f"> {pair[0]} vs. {pair[1]}")
                # sort the data by the group average
                data = self.df.copy()
                df = data.groupby(pair[1]).agg({pair[0]: "mean"}).reset_index()
                df = df.sort_values(by=pair[0]).reset_index(drop=True).reset_index()
                df = df.drop(columns=pair[0])
                data = data.merge(right=df, how="left", on=pair[1])
                data = data.sort_values(by="index").reset_index(drop=True)
                self.box_plot(
                    df=data, 
                    x=pair[0], 
                    y=pair[1],
                    title=f"{pair[0]} vs. {pair[1]}",
                    font_size=16,
                )

    def correlation_plot(self, df, title="Correlation Heatmap", font_size=None):
        df = df.copy()
        correlation = df.corr()

        # group columns together with hierarchical clustering
        X = correlation.values
        d = sch.distance.pdist(X)
        L = sch.linkage(d, method="ward")
        ind = sch.fcluster(L, 0.5*d.max(), "distance")
        columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
        df = df.reindex(columns, axis=1)
        
        # compute the correlation matrix for the received dataframe
        correlation = df.corr()

        # plot the correlation matrix
        fig = px.imshow(correlation, title=title, range_color=(-1, 1))
        fig.update_layout(font=dict(size=font_size), title_x=0.5)
        title = re.sub("[^A-Za-z0-9]+", " ", title)
        plot(fig, filename=f"{self.path}{path_sep}{self.name}{path_sep}{title}.html")

    def scatter_plot(self, df, x, y, color=None, title="Scatter Plot", font_size=None):
        fig = px.scatter(df, x=x, y=y, color=color, title=title)
        fig.update_layout(font=dict(size=font_size), title_x=0.5)
        title = re.sub("[^A-Za-z0-9]+", " ", title)
        plot(fig, filename=f"{self.path}{path_sep}{self.name}{path_sep}{title}.html")

    def bar_plot(self, df, x, y, color=None, title="Bar Plot", font_size=None):
        fig = px.bar(df, x=x, y=y, color=color, title=title)
        fig.update_layout(font=dict(size=font_size), title_x=0.5)
        title = re.sub("[^A-Za-z0-9]+", " ", title)
        plot(fig, filename=f"{self.path}{path_sep}{self.name}{path_sep}{title}.html")

    def box_plot(self, df, x, y, color=None, title="Box Plot", font_size=None):
        fig = px.box(df, x=x, y=y, color=color, title=title)
        fig.update_layout(font=dict(size=font_size), title_x=0.5)
        title = re.sub("[^A-Za-z0-9]+", " ", title)
        plot(fig, filename=f"{self.path}{path_sep}{self.name}{path_sep}{title}.html")

    def run_time(self, start, end):
        duration = end - start
        if duration < 60:
            duration = f"{round(duration, 2)} Seconds"
        elif duration < 3600:
            duration = f"{round(duration / 60, 2)} Minutes"
        else:
            duration = f"{round(duration / 3600, 2)} Hours"
        print(duration)

    def folder(self, name):
        if not os.path.isdir(name):
            os.mkdir(name)

