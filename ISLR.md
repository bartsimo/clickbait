**Prediction** In this setting, ˆf is often treated as a black box, in the sense
that one is not typically concerned with the exact form of ˆf , provided that
it yields accurate predictions for Y .

Why is the irreducible error larger than zero? The quantity ε may con-
tain unmeasured variables that are useful in predicting Y : since we don’t
measure them, f cannot use them for its prediction. The quantity ε may
also contain unmeasurable variation. For example, the risk of an adverse
reaction might vary for a given patient on a given day, depending on
manufacturing variation in the drug itself or the patient’s general feeling
of well-being on that day.

**Inference** We are often interested in understanding the association between Y and
X 1 , . . . , X p . In this situation we wish to estimate f , but our goal is not
necessarily to make predictions for Y . Now ˆf cannot be treated as a black
box, because we need to know its exact form. In this setting, one may be
interested in answering the following questions:
+ Which predictors are associated with the response? It is often the case
that only a small fraction of the available predictors are substantially
associated with Y . Identifying the few important predictors among a
large set of possible variables can be extremely useful, depending on
the application.
+ What is the relationship between the response and each predictor?
Some predictors may have a positive relationship with Y , in the sense
that larger values of the predictor are associated with larger values of
Y . Other predictors may have the opposite relationship. Depending
on the complexity of f , the relationship between the response and a
given predictor may also depend on the values of the other predictors.
+ Can the relationship between Y and each predictor be adequately sum-
marized using a linear equation, or is the relationship more compli-
cated? Historically, most methods for estimating f have taken a linear
form. In some situations, such an assumption is reasonable or even de-
sirable. But often the true relationship is more complicated, in which
case a linear model may not provide an accurate representation of
the relationship between the input and output variables.

Depending on whether our ultimate goal is prediction, inference, or a
combination of the two, different methods for estimating f may be appro-
priate. For example, linear models allow for relatively simple and inter- linear model
pretable inference, but may not yield as accurate predictions as some other
approaches. In contrast, some of the highly non-linear approaches that we
discuss in the later chapters of this book can potentially provide quite accu-
rate predictions for Y , but this comes at the expense of a less interpretable
model for which inference is more challenging.

## Training data
We will always assume that we have observed a set of n different
data points. For example in Figure 2.2 we observed n = 30 data points.
These observations are called the training data because we will use these observations to train, or teach, our method how to estimate f . Let x ij
represent the value of the jth predictor, or input, for observation i, where
i = 1, 2, . . . , n and j = 1, 2, . . . , p. Correspondingly, let y i represent the
response variable for the ith observation. Then our training data consist of
{(x 1 , y 1 ), (x 2 , y 2 ), . . . , (x n , y n )} where x i = (x i1 , x i2 , . . . , x ip ) T .
Our goal is to apply a statistical learning method to the training data
in order to estimate the unknown function f . In other words, we want to
find a function ˆf such that Y ≈ ˆf (X) for any observation (X, Y ). Broadly
speaking, most statistical learning methods for this task can be character-
ized as either parametric or non-parametric. We now briefly discuss these 
two types of approaches.

## Parametric approach / overfitting
The model-based approach just described is referred to as parametric;
it reduces the problem of estimating f down to one of estimating a set of
parameters. Assuming a parametric form for f simplifies the problem of
estimating f because it is generally much easier to estimate a set of pa-
rameters, such as β 0 , β 1 , . . . , β p in the linear model (2.4), than it is to fit
an entirely arbitrary function f . The potential disadvantage of a paramet-
ric approach is that the model we choose will usually not match the true
unknown form of f . If the chosen model is too far from the true f , then
our estimate will be poor. We can try to address this problem by choos-
ing flexible models that can fit many different possible functional forms 
for f . But in general, fitting a more flexible model requires estimating a
greater number of parameters. These more complex models can lead to a
phenomenon known as overfitting the data, which essentially means they 
follow the errors, or noise, too closely. These issues are discussed through- 
out this book.

## Non-parametric approach
Non-parametric methods do not make explicit assumptions about the func-
tional form of f . Instead they seek an estimate of f that gets as close to the
data points as possible without being too rough or wiggly. Such approaches
can have a major advantage over parametric approaches: by avoiding the
assumption of a particular functional form for f , they have the potential
to accurately fit a wider range of possible shapes for f . Any parametric
approach brings with it the possibility that the functional form used to
estimate f is very different from the true f , in which case the resulting
model will not fit the data well. In contrast, non-parametric approaches
completely avoid this danger, since essentially no assumption about the
form of f is made. But non-parametric approaches do suffer from a major
disadvantage: since they do not reduce the problem of estimating f to a
small number of parameters, a very large number of observations (far more
than is typically needed for a parametric approach) is required in order to
obtain an accurate estimate for f . Also: higher risk of overfitting.  
**Overfitting:** It is an undesirable situation because
the fit obtained will not yield accurate estimates of the response on new
observations that were not part of the original training data set.We have established that when inference is the goal, there are clear ad-
vantages to using simple and relatively inflexible statistical learning meth-
ods. In some settings, however, we are only interested in prediction, and
the interpretability of the predictive model is simply not of interest. For
instance, if we seek to develop an algorithm to predict the price of a
stock, our sole requirement for the algorithm is that it predict accurately—
interpretability is not a concern. In this setting, we might expect that it
will be best to use the most flexible model available. Surprisingly, this is
not always the case! We will often obtain more accurate predictions using
a less flexible method. This phenomenon, which may seem counterintuitive
at first glance, has to do with the potential for overfitting in highly flexible
methods

**Lasso** The lasso, discussed in Chapter 6, relies upon the lasso
linear model (2.4) but uses an alternative fitting procedure for estimating
the coefficients β 0 , β 1 , . . . , β p . The new procedure is more restrictive in es-
timating the coefficients, and sets a number of them to exactly zero. Hence
in this sense the lasso is a less flexible approach than linear regression.
It is also more interpretable than linear regression, because in the final
model the response variable will only be related to a small subset of the
predictors—namely, those with nonzero coefficient estimates.

Most statistical learning problems fall into one of two categories: supervised 
or unsupervised. The examples that we have discussed so far in this chapter all fall into the supervised learning domain. For each observation of the
predictor measurement(s) x i , i = 1, . . . , n there is an associated response
measurement y i . We wish to fit a model that relates the response to the
predictors, with the aim of accurately predicting the response for future
observations (prediction) or better understanding the relationship between
the response and the predictors (inference). Many classical statistical learn-
ing methods such as linear regression and logistic regression (Chapter 4), as well as more modern approaches such as GAM, boosting, and support vec-
tor machines, operate in the supervised learning domain. The vast majority
of this book is devoted to this setting.
By contrast, unsupervised learning describes the somewhat more chal-
lenging situation in which for every observation i = 1, . . . , n, we observe
a vector of measurements x i but no associated response y i . It is not pos-
sible to fit a linear regression model, since there is no response variable
to predict. In this setting, we are in some sense working blind; the sit-
uation is referred to as unsupervised because we lack a response vari-
able that can supervise our analysis. What sort of statistical analysis is
possible? We can seek to understand the relationships between the variables
or between the observations. One statistical learning tool that we may use in this setting is cluster analysis, or clustering.