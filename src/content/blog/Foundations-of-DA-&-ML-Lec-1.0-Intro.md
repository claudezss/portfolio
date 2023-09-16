---
title: "Foundations of DA & ML - Lec 1.0 Intro"
description: "Introduction of ML"
pubDate: "Sept 12 2023"
heroImage: "/blog.jpeg"
tags: ["AI&ML", "Lecture", "Foundations of DA & ML", "Python", "APS1070"] 
---


> This is a summary/knowledge base of my M.Eng. course content. <br>
> Please let me know if there is any infringement



## What is machine learning?

> Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy -- [IBM](https://www.ibm.com/topics/machine-learning)

Machine learning is an important component of the growing field of data science. Through the use of statistical methods, algorithms are trained to make classifications or predictions, and to uncover key insights in data mining projects. These insights subsequently drive decision making within applications and businesses, ideally impacting key growth metrics. As big data continues to expand and grow, the market demand for data scientists will increase. They will be required to help identify the most relevant business questions and the data to answer them. [[2]](#2)

Machine learning algorithms are typically created using frameworks that accelerate solution development, such as TensorFlow and PyTorch.[[1]](#1)

- A machine learning algorithm then takes “training data” and produces a model to generate the correct output
- If done correctly the program will generalize to cases not observed...more on this later
- **Instead of writing programs** by hand the **focus shifts to collecting quality examples** that highlight the correct output



### ML Applications [[1]](#1)

- ###### Voice recognition/synthesis
- AI assistant
- Machine translation
- Object recognition ( E.g:  [YOLO](https://pjreddie.com/darknet/yolo))
- Self-driving cars • Games
- Robots
- Face recognition • Healthcare
- Drug discovery
- ChatGPT
- Weather Forecasting/Prediction



---



## What is Data Science

> Data science combines math and statistics, specialized programming, advanced analytics, artificial intelligence (AI), and machine learning with specific subject matter expertise to uncover actionable insights hidden in an organization’s data. These insights can be used to guide decision making and strategic planning. -- [IBM](https://www.ibm.com/topics/data-science)

The data science lifecycle involves various roles, tools, and processes, which enables analysts to glean actionable insights. Typically, a data science project undergoes the following stages: [[3]](#3)

- **Data ingestion**: The lifecycle begins with the data collection--both raw structured and unstructured data from all relevant sources using a variety of methods. These methods can include manual entry, web scraping, and real-time streaming data from systems and devices. Data sources can include structured data, such as customer data, along with unstructured data like log files, video, audio, pictures, the Internet of Things (IoT), social media, and more.

- **Data storage and data processing:** Since data can have different formats and structures, companies need to consider different storage systems based on the type of data that needs to be captured. Data management teams help to set standards around data storage and structure, which facilitate workflows around analytics, machine learning and deep learning models. This stage includes cleaning data, deduplicating, transforming and combining the data using [ETL](https://www.ibm.com/topics/etl) (extract, transform, load) jobs or other data integration technologies. This data preparation is essential for promoting data quality before loading into a [data warehouse](https://www.ibm.com/topics/data-warehouse), [data lake](https://www.ibm.com/topics/data-lake), or other repository.

- **Data analysis:** Here, data scientists conduct an exploratory data analysis to examine biases, patterns, ranges, and distributions of values within the data. This data analytics exploration drives hypothesis generation for a/b testing. It also allows analysts to determine the data’s relevance for use within modeling efforts for predictive analytics, machine learning, and/or deep learning. Depending on a model’s accuracy, organizations can become reliant on these insights for business decision making, allowing them to drive more scalability.

- **Communicate:** Finally, insights are presented as reports and other data visualizations that make the insights—and their impact on business—easier for business analysts and other decision-makers to understand. A data science programming language such as R or Python includes components for generating visualizations; alternately, data scientists can use dedicated visualization tools.

Data Science is [[1]](#1):

- Multidisciplinary
- Digital revolution
- Data-driven discovery

It includes [[1]](#1):

- Data Mining
- Machine Learning
- Big Data
- Databases



### Data science versus data scientist

Data science is considered a discipline, while data scientists are the practitioners within that field. Data scientists are not necessarily directly responsible for all the processes involved in the data science lifecycle. For example, data pipelines are typically handled by data engineers—but the data scientist may make recommendations about what sort of data is useful or required. While data scientists can build machine learning models, scaling these efforts at a larger level requires more software engineering skills to optimize a program to run more quickly. As a result, it’s common for a data scientist to partner with machine learning engineers to scale machine learning models.

Data scientist responsibilities can commonly overlap with a data analyst, particularly with exploratory data analysis and data visualization. However, a data scientist’s skillset is typically broader than the average data analyst. Comparatively speaking, data scientist leverage common programming languages, such as R and Python, to conduct more statistical inference and data visualization.

To perform these tasks, data scientists require computer science and pure science skills beyond those of a typical business analyst or data analyst. The data scientist must also understand the specifics of the business, such as automobile manufacturing, eCommerce, or healthcare.

In short, a data scientist must be able to:

- Know enough about the business to ask pertinent questions and identify business pain points.
- Apply statistics and computer science, along with business acumen, to data analysis.
- Use a wide range of tools and techniques for preparing and extracting data—everything from databases and SQL to data mining to data integration methods.
- Extract insights from big data using predictive analytics and [artificial intelligence](https://www.ibm.com/topics/artificial-intelligence) (AI), including [machine learning models](https://www.ibm.com/blog/ai-vs-machine-learning-vs-deep-learning-vs-neural-networks/), [natural language processing](https://www.ibm.com/topics/natural-language-processing), and [deep learning](https://www.ibm.com/topics/deep-learning).
- Write programs that automate data processing and calculations.
- Tell—and illustrate—stories that clearly convey the meaning of results to decision-makers and stakeholders at every level of technical understanding.
- Explain how the results can be used to solve business problems.
- Collaborate with other data science team members, such as data and business analysts, IT architects, data engineers, and application developers.

These skills are in high demand, and as a result, many individuals that are breaking into a data science career, explore a variety of data science programs, such as certification programs, data science courses, and degree programs offered by educational institutions. [[3]](#3)



## Types of Machine Learning Systems

It is useful to classify machine learning systems into broad categories based on the following criteria: [[1]](#1)

- supervised, unsupervised, semi-supervised, and reinforcement learning
- classification versus regression
- online versus batch learning 
- instance-based versus model-based learning 
- parametric or nonparametric

### Supervised/Unsupervised Learning

Machine Learning systems can be classified according to the amount and type of supervision they get during training. [[1]](#1)

- Supervised

  k-Nearest Neighbours, Linear Regression, Logistic Regression, Decision Trees, Neural Networks, and many more

- Unsupervised
  K-Means, Principal Component Analysis

- Semi-supervised

- Reinforcement Learning

### Instance-Based/Model-Based Learning

#### Instance-Based

system learns the examples by heart, then generalizes to new cases by using a similarity/distance measure to compare them to the learned examples[[1]](#1)

#### Model-Based

build a model of these examples and then use that model to make predictions [[1]](#1)



## Challenges of Machine Learning [[1]](#1)

- Insufficient Data
- Quality Data
- Representative Data
- Irrelevant Features
- Overfitting the Training Data
- Underfitting the Training Data
- Testing and Validation
- Hyperparameter Tuning and Model Selection 
- Data Mismatch
- Fairness, Societal Concerns



## References
1. <a id="1"/>Uoft APS1070
2. <a id="2"/>IBM https://www.ibm.com/topics/machine-learning
3. <a id="3"/>IBM https://www.ibm.com/topics/data-science
