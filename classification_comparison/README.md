# Classifier Algorithms #

In a first instance, three selected algorithms were tested over a randomly generated dataset. This tiny analysis it's located in the basic classifier script.
After that, using a kaggle dataset, we tested the same set of algorithms

## Setup ##
```sh
    python3 -m venv env
    source env/bin/activate
    kaggle datasets download -d yersever/500-person-gender-height-weight-bodymassindex
```

### Problem Overiew ##

* Gender : Male / Female

* Height : Number (cm)

* Weight : Number (Kg)

Index :

0. Extremely Weak

1. Weak

2. Normal

3. Overweight

4. Obesity

5. Extreme Obesity