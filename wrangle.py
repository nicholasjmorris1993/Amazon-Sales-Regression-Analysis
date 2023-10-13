def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import re
import time
import numpy as np
import pandas as pd
import nltk
nltk.download("vader_lexicon")
from nltk.sentiment import SentimentIntensityAnalyzer
import scipy.cluster.hierarchy as sch
import plotly.express as px
from plotly.offline import plot

FIG_SIZE = (30, 30)  # size of the plots
FONT_SIZE = 16  # font size of text in plots

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

    def correlations(self):
        if self.plots:
            print("> Plotting Correlations")
            self.correlation_plot(
                df=self.df.drop(columns=["product_name", "category"]), 
                max_variables=None,
                title="Correlation Heatmap",
                font_size=16,
            )

    def scatter_plots(self):
        if self.plots:
            print("> Actual Price vs. Discount Percentage")
            self.scatter_plot(
                df=self.df,
                x="actual_price",
                y="discount_percentage",
                color=None,
                title="Actual Price vs. Discount Percentage",
                font_size=16,
            )

            print("> Actual Price vs. Rating")
            self.scatter_plot(
                df=self.df,
                x="actual_price",
                y="rating",
                color=None,
                title="Actual Price vs. Rating",
                font_size=16,
            )

            print("> Actual Price vs. Rating Count")
            self.scatter_plot(
                df=self.df,
                x="actual_price",
                y="rating_count",
                color=None,
                title="Actual Price vs. Rating Count",
                font_size=16,
            )

            print("> Actual Price vs. About Product Positivity")
            self.scatter_plot(
                df=self.df,
                x="actual_price",
                y="about_product_positivity",
                color=None,
                title="Actual Price vs. About Product Positivity",
                font_size=16,
            )

            print("> Actual Price vs. Review Title Positivity")
            self.scatter_plot(
                df=self.df,
                x="actual_price",
                y="review_title_positivity",
                color=None,
                title="Actual Price vs. Review Title Positivity",
                font_size=16,
            )

            print("> Actual Price vs. Review Content Positivity")
            self.scatter_plot(
                df=self.df,
                x="actual_price",
                y="review_content_positivity",
                color=None,
                title="Actual Price vs. Review Content Positivity",
                font_size=16,
            )
    
    def bar_plots(self):
        if self.plots:
            print("> Plotting Product Name")
            proportion = self.df["product_name"].value_counts(normalize=True).reset_index()
            proportion.columns = ["Label", "Proportion"]
            proportion = proportion.sort_values(by="Proportion", ascending=False).reset_index(drop=True)

            self.bar_plot(
                df=proportion,
                x="Proportion",
                y="Label",
                title="Product Name",
                font_size=16,
            )

            print("> Plotting Category")
            proportion = self.df["category"].value_counts(normalize=True).reset_index()
            proportion.columns = ["Label", "Proportion"]
            proportion = proportion.sort_values(by="Proportion", ascending=False).reset_index(drop=True)

            self.bar_plot(
                df=proportion,
                x="Proportion",
                y="Label",
                title="Category",
                font_size=16,
            )

    def pairwise_bar_plots(self):
        if self.plots:
            print("> Product Name vs. Category")
            data = self.df[["product_name", "category"]].copy()
            data["product_name, category"] = data["product_name"] + ", " + data["category"]
            proportion = data["product_name, category"].value_counts(normalize=True).reset_index()
            proportion.columns = ["Label", "Proportion"]
            proportion = proportion.sort_values(by="Proportion", ascending=False).reset_index(drop=True)

            self.bar_plot(
                df=proportion,
                x="Proportion",
                y="Label",
                title="Product Name vs. Category",
                font_size=16,
            )
    
    def boxplots(self):
        if self.plots:
            print("> Actual Price vs. Product Name")
            self.box_plot(
                df=self.df, 
                x="actual_price", 
                y="product_name",
                title="Actual Price vs. Product Name",
                font_size=16,
            )

            print("> Actual Price vs. Category")
            self.box_plot(
                df=self.df, 
                x="actual_price", 
                y="category",
                title="Actual Price vs. Category",
                font_size=16,
            )

    def correlation_plot(self, df, max_variables=None, title="Correlation Heatmap", font_size=None):
        df = df.copy()
        self.correlation = df.corr()

        # reduce the number of variables
        if max_variables is not None and df.shape[1] > max_variables:
            # flatten the correlation matrix into pairings
            correlations = self.correlation.copy()
            correlations = correlations.stack().reset_index()
            correlations.columns = ["Variable 1", "Variable 2", "Correlation"]

            # remove duplicate pairings
            mask_duplicates = (correlations[["Variable 1", "Variable 2"]].apply(frozenset, axis=1).duplicated()) | (correlations["Variable 1"]==correlations["Variable 2"]) 
            correlations = correlations[~mask_duplicates]

            # only keep correlations with a magnitude of at least 0.8
            correlations["Magnitude"] = correlations["Correlation"].abs()
            correlations = correlations.sort_values(by="Magnitude", ascending=False).reset_index(drop=True)
            correlations = correlations.loc[correlations["Magnitude"] >= 0.8]

            # limit the number of pairings to plot
            correlations = correlations.head(max_variables)

            # choose which variables to keep
            variables = list()
            for i in range(correlations.shape[0]):
                variables.append(correlations["Variable 1"][i])
                variables.append(correlations["Variable 2"][i])
            variables = pd.unique(variables)[:max_variables]

            df = df[variables]
            self.correlation = df.corr()

        # group columns together with hierarchical clustering
        X = self.correlation.values
        d = sch.distance.pdist(X)
        L = sch.linkage(d, method="ward")
        ind = sch.fcluster(L, 0.5*d.max(), "distance")
        columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
        df = df.reindex(columns, axis=1)
        
        # compute the correlation matrix for the received dataframe
        self.correlation = df.corr()

        # plot the correlation matrix
        fig = px.imshow(self.correlation, title=title)
        fig.update_layout(font=dict(size=font_size))
        title = re.sub("[^A-Za-z0-9]+", " ", title)
        plot(fig, filename=f"{self.path}{path_sep}{self.name}{path_sep}{title}.html")

    def scatter_plot(self, df, x, y, color=None, title="Scatter Plot", font_size=None):
        fig = px.scatter(df, x=x, y=y, color=color, title=title)
        fig.update_layout(font=dict(size=font_size))
        title = re.sub("[^A-Za-z0-9]+", " ", title)
        plot(fig, filename=f"{self.path}{path_sep}{self.name}{path_sep}{title}.html")

    def bar_plot(self, df, x, y, color=None, title="Bar Plot", font_size=None):
        fig = px.bar(df, x=x, y=y, color=color, title=title)
        fig.update_layout(font=dict(size=font_size))
        title = re.sub("[^A-Za-z0-9]+", " ", title)
        plot(fig, filename=f"{self.path}{path_sep}{self.name}{path_sep}{title}.html")

    def box_plot(self, df, x, y, color=None, title="Box Plot", font_size=None):
        fig = px.box(df, x=x, y=y, color=color, title=title)
        fig.update_layout(font=dict(size=font_size))
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

