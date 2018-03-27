<div style="text-align:center"><img src ="http://i.imgur.com/dI2Q3hn.png" /></div>

# Machine Learning Short Information

Machine Learning is a branch of Artificial Intelligence dedicated at making
machines learn from observational data without being explicitly programmed. 

> Machine learning and AI are not the same. Machine learning is an instrument in
> the AI symphony — a component of AI. So what is Machine Learning — or ML —
> exactly? It’s the ability for an algorithm to learn from prior data in order
> to produce a behavior. ML is teaching machines to make decisions in situations
> they have never seen.


## Machine Learning in General

Study this section to understand fundamental concepts and develop intuitions before going any deeper.

> A computer program is said to learn from experience `E` with respect to some
> class of tasks `T` and performance measure `P` if its performance at tasks in
> `T`, as measured by `P`, improves with experience `E`.

* [Artificial Intelligence, Revealed](https://code.facebook.com/pages/1902086376686983) a quick introduction by Yann LeCun, mostly about Machine Learning ideas, Deep Learning, and convolutional neural network
* [How do I learn machine learning? - Quora](https://www.quora.com/How-do-I-learn-machine-learning-1)
* [Intro to Machine Learning - Udacity](https://www.udacity.com/course/intro-to-machine-learning--ud120) hands on scikit-learn (python) programming learning on core ML concepts
* [Machine Learning: Supervised, Unsupervised & Reinforcement - Udacity](https://www.udacity.com/course/machine-learning--ud262) the 2 instructors are hilarious
* [Machine Learning Mastery](http://machinelearningmastery.com/start-here/) very carefully laid out step-by-step guide to some particular algorithms, though contains inaccuracies about general principles such as the bias-variance tradeoff.
* [Andrew Ng's Course on Coursera](https://www.coursera.org/learn/machine-learning) recommended for people who want to know the details of ML algorithms under the hood, understand enough maths to be dangerous and do coding assignments in Octave programming language (Note: Andrew said that you do not need to know Calculus beforehand but he talked about it quite often so you will regret if you don't know Calculus :laughing:)
* [ML Recipes - YouTube Playlist](https://www.youtube.com/playlist?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal) a really nicely designed concrete actionable content for ML introduction
* [Machine Learning is Fun Part 1](https://medium.com/@ageitgey/machine-learning-is-fun-80ea3ec3c471) simple approach to machine learning for non-maths people
* [Machine Learning with Python - YouTube Playlist](https://www.youtube.com/playlist?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v)
* [Machine Learning Yearning by Andrew Ng](http://www.mlyearning.org/)
* [Machine Learning Crash Course: Part 1](https://ml.berkeley.edu/blog/2016/11/06/tutorial-1/)
* [https://www.kadenze.com/courses/machine-learning-for-musicians-and-artists-iv](https://www.kadenze.com/courses/machine-learning-for-musicians-and-artists-iv)
* [Rules of Machine Learning: Best Practices for ML Engineering](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf)
* [Most Shared Machine Learning Content on Twitter For The Past 7 Days](http://theherdlocker.com/tweet/popularity/machinelearning)
* [MIT 6.S099: Artificial General Intelligence](https://agi.mit.edu/) This class takes an engineering approach to exploring possible research paths toward building human-level intelligence.


## Deep Learning

Deep learning is a branch of machine learning where deep artificial neural
networks (DNN) — algorithms inspired by the way neurons work in the brain — find
patterns in raw data by combining multiple layers of artificial neurons. As the
layers increase, so does the neural network’s ability to learn increasingly
abstract concepts.

The simplest kind of DNN is
a [Multilayer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)
(MLP).

<div style="text-align:center"><img src ="https://github.com/off99555/machine-learning-curriculum/blob/master/img/Brain-549603035-1.jpg" /></div>

# Supervised learning algorithms 

Attention: In all supervised learning algorithms used the same data.

___
[Naive Bayes code samples](https://github.com/narekye/machine-learning/tree/master/udacity/udacity/naive_bayes)
> Naive Bayes documentation [sklearn package](http://scikit-learn.org/stable/modules/naive_bayes.html)

<div style="text-align:center">
  <img src="https://github.com/narekye/machine-learning/blob/master/udacity/udacity/data/naive_bayes_classifier.png" /><
  /div>

Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of independence between every pair of features. Given a class variable y and a dependent feature vector x_1 through x_n, Bayes’ theorem states the following relationship:

Naive Bayes learners and classifiers can be extremely fast compared to more sophisticated methods. The decoupling of the class conditional feature distributions means that each distribution can be independently estimated as a one dimensional distribution. This in turn helps to alleviate problems stemming from the curse of dimensionality.
