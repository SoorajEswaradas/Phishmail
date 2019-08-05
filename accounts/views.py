from django.shortcuts import render, redirect
import subprocess
from django.contrib.auth import (
    authenticate,
    get_user_model,
    login,
    logout
)

from django.contrib.auth.models import User


from .models import mails

from .forms import UserLoginForm, UserRegisterForm


def login_view(request):
    next = request.GET.get('next')
    form = UserLoginForm(request.POST or None)
    if form.is_valid():
        username = form.cleaned_data.get('username')
        password = form.cleaned_data.get('password')
        user = authenticate(username=username, password=password)
        login(request, user)
        if next:
            return redirect(next)
        return redirect('/')
    context = {
        'form':  form
    }
    return render(request, "login.html", context)

def register_view(request):
    next = request.GET.get('next')
    form = UserRegisterForm(request.POST or None)
    if form.is_valid():
        user = form.save(commit=False)
        password = form.cleaned_data.get('password')
        user.set_password(password)
        user.save()
        new_user = authenticate(username=user.username, password=password)
        login(request, new_user)
        if next:
            return redirect(next)
        return redirect('/')
    context = {
        'form':  form
    }
    return render(request, "signup.html", context)

def new_page(request):

    user_email = request.user.email
    if request.GET["email"] and request.GET["sub"] and request.GET["body"]:
        data = request.GET["email"]
        subject = request.GET["sub"]
        body = request.GET["body"]

      ########################
        from sklearn import tree
        from sklearn.metrics import accuracy_score
        from IPython.display import Image
        import numpy as np
        import io
        import pydotplus

        from graphviz import Source

        import regex as re
        myString = body
        print(re.search("(?P<url>https?://[^\s]+)", myString).group("url"))
        url = re.search("(?P<url>https?://[^\s]+)", myString).group("url")

        arr = []
        arr.append([])

        print(arr)

        # Method to count number of dots
        def countdots(url):
            print("Dots = ", url.count('.'))

        # Method to count number of delimeters
        def countdelim(url):
            count = 0
            delim = [';', '_', '?', '=', '&']
            for each in url:
                if each in delim:
                    count = count + 1
            print("Delimiters = ", count)

        def urllength(url):
            if len(url) < 54:
                arr[0].append(-1)
            elif len(url) >= 54 and len(url) <= 75:
                arr[0].append(0)
            else:
                arr[0].append(1)
            print("Length = ", len(url))

        # Is IP addr present as th hostname, let's validate

        import ipaddress as ip  # works only in python 3

        def isip(uri):
            try:
                if ip.ip_address(uri):
                    arr[0].append(1)
                    print("IP present")
            except:
                arr[0].append(-1)
                print("No ip present")

        # method to check the presence of hyphens

        def isPresentHyphen(url):
            if url.count('-') >= 1:
                arr[0].append(1)
            else:
                arr[0].append(-1)

            print("Number of hiphens = ", url.count('-'))

        # method to check the presence of @

        def isPresentAt(url):
            if url.count('@') >= 1:
                arr[0].append(1)
            else:
                arr[0].append(-1)
            print("Number of @ = ", url.count('@'))

        def isPresentDSlash(url):
            if url.count('//') >= 2:
                arr[0].append(1)
            else:
                arr[0].append(-1)

            print("Number of // = ", url.count('//'))

        def countSubDir(url):
            return url.count('/')

        def get_ext(url):
            """Return the filename extension from url, or ''."""

            root, ext = splitext(url)
            return ext

        def countSubDomain(subdomain):
            if url.count('.') >= 3:
                arr[0].append(1)
            elif url.count('.') >= 2:
                arr[0].append(0)
            else:
                arr[0].append(-1)

        def countQueries(query):
            if not query:
                return 0
            else:
                return len(query.split('&'))

        def httpsstart(url):
            if (url.startswith("https")):
                arr[0].append(-1)
            else:
                arr[0].append(1)

        def httpsindomain(url):
            if (url.count("https") > 1):
                arr[0].append(1)
            else:
                arr[0].append(-1)

        isip(url)
        urllength(url)
        isPresentAt(url)
        isPresentDSlash(url)
        isPresentHyphen(url)
        countSubDomain(url)
        httpsstart(url)
        httpsindomain(url)

        print(arr)
        training_data = np.genfromtxt('\\Users\\Admin\\Desktop\\PycharmProjects\\userauth\\phishmail\\accounts\\dataset.csv', delimiter=',', dtype=np.int32)
        def load_data():
            """
            This helper function loads the dataset saved in the CSV file
            and returns 4 numpy arrays containing the training set inputs
            and labels, and the testing set inputs and labels.
            """

            # Load the training data from the CSV file
            #training_data = np.genfromtxt('dataset.csv', delimiter=',', dtype=np.int32)

            """
            Each row of the CSV file contains the features collected on a website
            as well as whether that website was used for phishing or not.
            We now separate the inputs (features collected on each website)
            from the output labels (whether the website is used for phishing).
            """

            # Extract the inputs from the training data array (all columns but the last one)
            inputs = training_data[:, :-23]

            # Extract the outputs from the training data array (last column)
            outputs = training_data[:, -1]

            # Separate the training (first 2,000 websites) and testing data (last 456)
            training_inputs = inputs[:2000]
            training_outputs = outputs[:2000]
            testing_inputs = arr

            # testing_outputs = outputs[2000:]

            # Return the four arrays
            return training_inputs, training_outputs, testing_inputs  # , testing_outputs

        #if __name__ == '__main__':
        if 1:
            print("Tutorial: Training a decision tree to detect phishing websites")

            # Load the training data
            train_inputs, train_outputs, test_inputs = load_data()
            print("Training data loaded.")

            # Create a decision tree classifier model using scikit-learn
            classifier = tree.DecisionTreeClassifier()
            print("Decision tree classifier created.")
            print(classifier)

            print("Beginning model training.")
            # Train the decision tree classifier
            classifier.fit(train_inputs, train_outputs)
            print("Model training completed.")

            # Use the trained classifier to make predictions on the test data
            predictions = classifier.predict(test_inputs)
            print("Predictions on testing data computed.")
            if predictions == 1:
                print("Phishing")
                print(predictions)
            else:
                print("Safe")
                print(predictions)


                # Print the accuracy (percentage of phishing websites correctly predicted)
            # accuracy = 100.0 * accuracy_score(test_outputs, predictions)
            # print("The accuracy of your decision tree on testing data is: " + str(accuracy))


        from sklearn.linear_model import LogisticRegression as lr
        from sklearn.metrics import accuracy_score
        import numpy as np

        #from decision_tree import load_data

        #if __name__ == '__main__':
        if 1:
            print("Tutorial: Training a logistic regression to detect phishing websites")

            # Load the training data
            train_inputs, train_outputs, test_inputs = load_data()
            print("Training data loaded.")

            # Create a logistic regression classifier model using scikit-learn
            classifier = lr()
            print("Logistic regression classifier created.")

            print("Beginning model training.")
           # Train the logistic regression classifier
            classifier.fit(train_inputs, train_outputs)
            print("Model training completed.")

            test_inputs = arr

                # Use the trained classifier to make predictions on the test data
            predictions = classifier.predict(test_inputs)
            print("Predictions on testing data computed.")
            print("Result", predictions)
                # Print the accuracy (percentage of phishing websites correctly predicted)
                # accuracy = 100.0 * accuracy_score(test_outputs, predictions)
                # print("The accuracy of your logistic regression on testing data is: " + str(accuracy))

        ##################

        if data and subject and body:
            mailvar = mails()
            mailvar.frommail = user_email
            mailvar.to = data
            mailvar.subject = subject
            mailvar.body = body
            mailvar.save()

    refer1 = reversed(mails.objects.all())
    refer2 = reversed(mails.objects.all())
    return render(request, 'home.html', {'refer1': refer1, 'refer2': refer2, 'userm': user_email})



def logout_view(request):
        logout(request)
        return redirect('/')
