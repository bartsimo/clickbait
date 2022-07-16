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
observations that were not part of the original training data set.