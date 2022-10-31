"# Data-Science-Machine-Learning-with-Python"

## ID3 Algorithm in Decision Tree :
   ID3 algorithm stands for Iterative Dichotomiser 3, which is a classification algorithm that follows a greedy approach of building a decision tree by selecting the best attribute that yields maximum Information Gain (IG) or minimum Entropy (H).

We will go through the basics of the decision tree, ID3 algorithm before applying it to our data. We will answer the following questions:

1. What is a decision tree?
2. What is the ID3 algorithm?
3. What is information gain and entropy?
4. What are the steps in the ID3 algorithm?
5. Using ID3 algorithm on a real data
6. What are the characteristics of the ID3 algorithm?

What is a Decision Tree?
A Supervised Machine Learning Algorithm, used to build classification and regression models in the form of a tree structure.

A decision tree is a tree where each -

Node - a feature(attribute)
Branch - a decision(rule)
Leaf - an outcome(categorical or continuous)
There are many algorithms to build decision trees, here we are going to discuss the ID3 algorithm with an example.

What is an ID3 Algorithm?
ID3 stands for Iterative Dichotomiser 3

It is a classification algorithm that follows a greedy approach by selecting the best attribute that yields maximum Information Gain(IG) or minimum Entropy(H).

What are Entropy and Information gain?
Entropy is a measure of the amount of uncertainty in the dataset S. Mathematical Representation of Entropy is shown here -

H ( S ) = ∑ c ∈ C − p ( c ) l o g 2 p ( c )
Where,

S - The current dataset for which entropy is being calculated(changes every iteration of the ID3 algorithm).
C - Set of classes in S {example - C ={yes, no}}
p(c) - The proportion of the number of elements in class c to the number of elements in set S.
In ID3, entropy is calculated for each remaining attribute. The attribute with the smallest entropy is used to split the set S on that particular iteration.
Entropy = 0 implies it is of pure class, which means all are of the same category.

Information Gain IG(A) tells us how much uncertainty in S was reduced after splitting set S on attribute A. Mathematical representation of Information gain is shown here -

I G ( A , S ) = H ( S ) − ∑ t ∈ T p ( t ) H ( t )
Where,

H(S) - Entropy of set S.
T - The subsets created from splitting set S by attributing A such that
S = ⋃ t ϵ T t
p(t) - The proportion of the number of elements int to the number of elements in set S.
H(t) - Entropy of subset t.
In ID3, information gain can be calculated (instead of entropy) for each remaining attribute. The attribute with the largest information gain is used to split the set S on that particular iteration.

What are the steps in the ID3 algorithm?

Calculate entropy for the dataset.
For each attribute/feature.
Calculate entropy for all its categorical values.
Calculate information gain for the feature.
Find the feature with maximum information gain.
Repeat it until we get the desired tree.
