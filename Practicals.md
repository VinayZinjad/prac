## Slip 1
1. Write a Java Program to implement I/O Decorator for converting uppercase letters to lower case letters.
```java 
//Main.java
import java.io.*;
import java.io.IOException;

public class Main {

    public static void main(String[] args) {
        try {
            InputStream inputStream = new LowerCaseInputStream(new FileInputStream("example.txt"));
            int value;
            while ((value = inputStream.read()) != -1) {
                System.out.print((char) value);
            }
            inputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

```java
//LowerCaseInputStream.java
import java.io.IOException;
import java.io.InputStream;

public class LowerCaseInputStream extends InputStream {

    private InputStream inputStream;

    public LowerCaseInputStream(InputStream inputStream) {
        this.inputStream = inputStream;
    }

    @Override
    public int read() throws IOException {
        int value = inputStream.read();
        if (value >= 'A' && value <= 'Z') {
            value = value + 'a' - 'A';
        }
        return value;
    }
}
```

2. Write a python program to Prepare Scatter Plot (Use Forge Dataset / Iris Dataset)

```python
#scatter.ipynb
import pandas as pd
import matplotlib.pyplot as plt
iris = pd.read_csv("Iris1.csv") 
iris.head()
iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")


#OR

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()

# Create a dataframe
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Create a new column for target names
df['Target'] = iris.target_names[iris.target]

# Display the dataframe
print(df.head())

# Prepare the scatter plot
for target_name, target in zip(iris.target_names, range(iris.target_names.size)):
    indicesToKeep = iris.target == target
    plt.scatter(iris.data[indicesToKeep, 0], iris.data[indicesToKeep, 1], label=target_name)

plt.legend()
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Scatter Plot')
plt.show()
```

3. Student form
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Student Registration</title>
  </head>
  <body>
    <h1>Student Registration</h1>
    <label for="firstName">First Name</label>
    <input type="text" id="firstName" name="firstName" />
    <br />
    <label for="lastName">Last Name</label>
    <input type="text" id="lastName" name="lastName" />
    <br />
    <label for="age">Age</label>
    <input type="text" id="age" name="age" />
    <br />
    <button id="submit" type="button">Submit</button>
  </body>

  <script>
    const firstName = document.getElementById("firstName");
    const lastName = document.getElementById("lastName");
    const age = document.getElementById("age");
    const submit = document.getElementById("submit");

    function validateInput() {
      let firstNameValue = firstName?.value;
      let lastNameValue = lastName?.value;
      let ageValue = age?.value;

      console.log(firstNameValue);
      console.log(lastNameValue);
      console.log(ageValue);

      if (firstNameValue === "") {
        alert("First Name is required.");
        return false;
      }

      if (lastNameValue === "") {
        alert("Last Name is required.");
        return false;
      }

      if (ageValue === "") {
        alert("Age is required.");
        return false;
      }

      if (!firstNameValue?.match(/^[a-zA-Z]+$/)) {
        alert("First Name can only contain alphabets.");
        return false;
      }

      if (!lastNameValue?.match(/^[a-zA-Z]+$/)) {
        alert("Last Name can only contain alphabets.");
        return false;
      }

      if (!ageValue?.match(/^\d+$/)) {
        alert("Age can only contain numbers.");
        return false;
      }
      if (ageValue < 19 || ageValue > 49) {
        alert("Age should be between 18 and 50.");
        return false;
      }

      return true;
    }
    submit.addEventListener("click", function () {
      if (validateInput()) {
        alert("Student Registration Successful.");
      }
    });
  </script>
</html>

```


---

## Slip 2

1. Write a Java program to implement singleton pattern for multithreading

```java
//Main.java
public class Main {
    public static void main(String[] args) throws Exception {
        Test1 t = Test1.getInstance();
        Test1 t1 = Test1.getInstance();
        Test1 t2 = Test1.getInstance();
        Test1 t3 = Test1.getInstance();
        Test1 t4 = Test1.getInstance();

        Thread tt = new Thread(t);
        Thread tt2 = new Thread(t2);
        Thread tt3 = new Thread(t3);
        Thread tt4 = new Thread(t);
        Thread tt5 = new Thread(t);

        tt.start();
        tt2.start();
        tt3.start();
        tt4.start();
        tt5.start();
    }
}
```

```java
//Test1.java

import java.util.Random;

public class Test1 implements Runnable {

    private static Test1 instance = null;
    private final Random random = new Random();

    private Test1() {
    }

    public static Test1 getInstance() {
        if (instance == null) {
            instance = new Test1();
        }
        return instance;
    }

    @Override
    public void run() {
        int value = random.nextInt(100);
        System.out.println(Thread.currentThread().getName() + " generated: " + value);
    }
}
```

2. Write a python program the Categorical values in numeric format for a given dataset

```python
#catagorical
# importing pandas as pd
import pandas as pd
#importing data using .read_csv() function
df = pd.read_csv('DataMLcategorical2.csv')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Using .fit_transform function to fit label
# encoder and return encoded label
label = le.fit_transform(df['Purchased'])

# printing label
label

# as it is of no use now.
df.drop("Purchased", axis=1, inplace=True)
# Appending the array to our dataFrame
# with column name 'Purchased'
df["Purchased"] = label
# printing Dataframe
df
```

3. Create a Node.js file that will convert the output "Hello World!" into upper-case letters

```js
var http = require('http');
var uc = require('upper-case');
http.createServer(function (req, res) {
 res.writeHead(200, {'Content-Type': 'text/html'});
 /*Use our upper-case module to upper case a string:*/
 res.write(uc.upperCase("Hello World!"));
 res.end();
}).listen(8080);
```


--- 

## Slip 3

1.Write a java program to create shape interface and concreate classes implementing shape interface (circle square rectangle). Use this code to implement factory method

```java
// Shape.java
public interface Shape {
    void draw();
}

// Circle.java
public class Circle implements Shape {
    @Override
    public void draw() {
        System.out.println("Drawing a Circle");
    }
}

// Square.java
public class Square implements Shape {
    @Override
    public void draw() {
        System.out.println("Drawing a Square");
    }
}

// Rectangle.java
public class Rectangle implements Shape {
    @Override
    public void draw() {
        System.out.println("Drawing a Rectangle");
    }
}

// ShapeFactory.java
public class ShapeFactory {
    public static Shape createShape(String shapeType) {
        if (shapeType.equalsIgnoreCase("Circle")) {
            return new Circle();
        } else if (shapeType.equalsIgnoreCase("Square")) {
            return new Square();
        } else if (shapeType.equalsIgnoreCase("Rectangle")) {
            return new Rectangle();
        } else {
            throw new IllegalArgumentException("Invalid shape type");
        }
    }
}

// Main.java
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.print("Enter shape type (Circle/Square/Rectangle): ");
        String shapeType = scanner.nextLine();

        // Use factory method to create a shape
        Shape shape = ShapeFactory.createShape(shapeType);

        // Draw the shape
        shape.draw();
    }
}

```

2. Write a python program to implement simple Linear Regression for predicting salary.

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Salary_DataSimpleLinearRegression.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```

3. .Using nodejs create a web page to read two file names from user and append contents of the first file into the second file
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Document</title>
  </head>
  <body>
    <form action="/appendFiles" method="POST">
      <label for="firstFileName">First File Name</label>
      <input type="text" name="firstFileName" id="firstFileName" />
      <br />
      <label for="secondFileName">Second File Name</label>
      <input type="text" name="secondFileName" id="secondFileName" />
      <br />
      <button>Append Files</button>
    </form>
  </body>
</html>

```

```js
const http = require("http");
const fs = require("fs");
const url = require("url");
const path = require("path");

const server = http.createServer((req, res) => {
    const { pathname, query } = url.parse(req.url, true);

    if (pathname == "/") {
        //Server HTML

        res.writeHead(200, { "Content-Type": "text/html" });
        fs.createReadStream("append-files.html").pipe(res);
    } else if (pathname === "/appendFiles" && req.method === "POST") {
        let body = "";

        req.on("data", (chunk) => {
            body += chunk;
        });

        req.on("end", () => {
            const formData = new URLSearchParams(body);
            const firstFileName = formData.get("firstFileName");
            const secondFileName = formData.get("secondFileName");

            // read the content

            fs.readFile(firstFileName, "utf-8", (err, data) => {
                if (err) {
                    res.writeHead(500, { ContentType: "text/plain" });
                    res.end("Error reading files.");
                } else {
                    fs.appendFile(secondFileName, data, (err) => {
                        if (err) {
                            res.writeHead(500, { ContentType: "text/plain" });
                            res.end("Error reading files.");
                        } else {
                            res.writeHead(200, { ContentType: "text/plain" });
                            res.end("updated.");
                        }
                    });
                }
            });
        });
    }
});
const port = 8081;
server.listen(port, () => {
    console.log(`server listening on ${port}`);
});
```

---
## Slip 4

1. Write a Java Program to implement Abstract Factory Pattern for Shape interface

```java
// Shape.java
public interface Shape {
    void draw();
}

// Circle.java
public class Circle implements Shape {
    @Override
    public void draw() {
        System.out.println("Drawing a Circle");
    }
}

// Square.java
public class Square implements Shape {
    @Override
    public void draw() {
        System.out.println("Drawing a Square");
    }
}

// Rectangle.java
public class Rectangle implements Shape {
    @Override
    public void draw() {
        System.out.println("Drawing a Rectangle");
    }
}

// AbstractFactory.java
public interface AbstractFactory {
    Shape createCircle();
    Shape createSquare();
    Shape createRectangle();
}

// ShapeFactory.java
public class ShapeFactory implements AbstractFactory {
    @Override
    public Shape createCircle() {
        return new Circle();
    }

    @Override
    public Shape createSquare() {
        return new Square();
    }

    @Override
    public Shape createRectangle() {
        return new Rectangle();
    }
}

// Main.java
public class Main {
    public static void main(String[] args) {
        AbstractFactory factory = new ShapeFactory();

        Shape circle = factory.createCircle();
        circle.draw();

        Shape square = factory.createSquare();
        square.draw();

        Shape rectangle = factory.createRectangle();
        rectangle.draw();
    }
}

```

2.  Write a python program to implement Polynomial Regression for given dataset

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Position_SalariesPolynomialRegression.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
```

3. Create an HTML form for Login and write a JavaScript to validate email ID using Regular Expression
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>JavaScript form validation - checking email</title>
    <link rel="stylesheet" href="form-style.css" type="text/css" />
  </head>
  <body onload="document.form1.text1.focus()">
    <div class="mail">
      <h2>Input an email and Submit</h2>
      <form name="form1" action="#">
        <label for="text1">Email</label>
        <input type="text" name="text1" />
        <br />
        <label for="text2">Password</label>
        <input type="password" name="text2" />
        <br />

        <input
          type="submit"
          name="submit"
          value="Submit"
          onclick="ValidateEmail(document.form1.text1)"
        />
      </form>
    </div>
    <script>
      function ValidateEmail(inputText) {
        var mailformat = /^\S+@\S+\.\S+$/;
        if (inputText.value.match(mailformat)) {
          alert("Valid email address!");
          document.form1.text1.focus();
          return true;
        } else {
          alert("You have entered an invalid email address!");
          document.form1.text1.focus();
          return false;
        }
      }
    </script>
  </body>
</html>
```


---
## Slip 5

1. Write a Java Program to implement Decorator Pattern

```java

// Component interface
interface Coffee {
    double cost();

    String description();
}

// Concrete component
class SimpleCoffee implements Coffee {
    @Override
    public double cost() {
        return 2.0;
    }

    @Override
    public String description() {
        return "Simple Coffee";
    }
}

// Decorator abstract class
abstract class CoffeeDecorator implements Coffee {
    protected Coffee decoratedCoffee;

    public CoffeeDecorator(Coffee decoratedCoffee) {
        this.decoratedCoffee = decoratedCoffee;
    }

    @Override
    public double cost() {
        return decoratedCoffee.cost();
    }

    @Override
    public String description() {
        return decoratedCoffee.description();
    }
}

// Concrete decorators
class MilkDecorator extends CoffeeDecorator {
    public MilkDecorator(Coffee decoratedCoffee) {
        super(decoratedCoffee);
    }

    @Override
    public double cost() {
        return super.cost() + 0.5;
    }

    @Override
    public String description() {
        return super.description() + " with Milk";
    }
}

class SugarDecorator extends CoffeeDecorator {
    public SugarDecorator(Coffee decoratedCoffee) {
        super(decoratedCoffee);
    }

    @Override
    public double cost() {
        return super.cost() + 0.2;
    }

    @Override
    public String description() {
        return super.description() + " with Sugar";
    }
}

// Main class
public class Main {
    public static void main(String[] args) {
        // Create a simple coffee
        Coffee simpleCoffee = new SimpleCoffee();
        System.out.println("Cost: $" + simpleCoffee.cost() + ", Description: " + simpleCoffee.description());

        // Decorate the coffee with milk
        Coffee milkCoffee = new MilkDecorator(simpleCoffee);
        System.out.println("Cost: $" + milkCoffee.cost() + ", Description: " + milkCoffee.description());

        // Decorate the coffee with sugar
        Coffee sugarCoffee = new SugarDecorator(simpleCoffee);
        System.out.println("Cost: $" + sugarCoffee.cost() + ", Description: " + sugarCoffee.description());

        // Decorate the coffee with both milk and sugar
        Coffee milkAndSugarCoffee = new SugarDecorator(new MilkDecorator(simpleCoffee));
        System.out
                .println("Cost: $" + milkAndSugarCoffee.cost() + ", Description: " + milkAndSugarCoffee.description());
    }
}

```

2. Write a python program to Implement Naïve Bayes
```python
from sklearn.datasets import load_iris
iris = load_iris()
# store the feature matrix (X) and response vector (y)
X = iris.data
y = iris.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
```

3. Create a node js file that opens the requested file and returns content to the client. if anything goes wrong, throw a 404 error

```js
const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

const server = http.createServer((req, res) => {
    const reqUrl = url.parse(req.url, true);
    const pathname = reqUrl.pathname === '/' ? '/index.html' : reqUrl.pathname;

    if (req.method === 'GET') {
        if (pathname === '/index.html') {
            // Serve HTML page with input field
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.end(`
        <!DOCTYPE html>
        <html>
          <head>
            <title>File Reader</title>
          </head>
          <body>
            <h1>Enter filename:</h1>
            <form action="/" method="POST">
              <input type="text" name="filename" />
              <input type="submit" value="Read File" />
            </form>
          </body>
        </html>
      `);
        } else {
            // Handle other requests
            res.writeHead(404, { 'Content-Type': 'text/plain' });
            res.end('Not Found');
        }
    } else if (req.method === 'POST' && reqUrl.pathname === '/') {
        // Handle form submission
        let body = '';
        req.on('data', (chunk) => {
            body += chunk;
        });

        req.on('end', () => {
            const filename = decodeURIComponent(body.split('=')[1]);
            const filePath = path.join(__dirname, filename);

            fs.readFile(filePath, 'utf8', (err, data) => {
                if (err) {
                    res.writeHead(404, { 'Content-Type': 'text/plain' });
                    res.end('File not found');
                } else {
                    res.writeHead(200, { 'Content-Type': 'text/html' });
                    res.end(`
            <!DOCTYPE html>
            <html>
              <head>
                <title>File Reader</title>
              </head>
              <body>
                <h1>File Content:</h1>
                <pre>${data}</pre>
              </body>
            </html>
          `);
                }
            });
        });
    }
});

const PORT = 3000;
server.listen(PORT, () => {
    console.log(`Server is running at http://localhost:${PORT}`);
});

```


## Slip 6

1. Write a Java Program to implement Facade Design Pattern.

```java
// Subsystem class 1
class Subsystem1 {
    public void operation1() {
        System.out.println("Subsystem1: Operation 1");
    }
}

// Subsystem class 2
class Subsystem2 {
    public void operation2() {
        System.out.println("Subsystem2: Operation 2");
    }
}

// Facade class
class Facade {
    private Subsystem1 subsystem1;
    private Subsystem2 subsystem2;

    public Facade() {
        this.subsystem1 = new Subsystem1();
        this.subsystem2 = new Subsystem2();
    }

    // Facade method that simplifies the operations
    public void doSomething() {
        System.out.println("Facade: Doing something...");
        subsystem1.operation1();
        subsystem2.operation2();
    }
}

// Client class
public class FacadePatternExample {
    public static void main(String[] args) {
        // Using the Facade to simplify the subsystem operations
        Facade facade = new Facade();
        facade.doSomething();
    }
}

```

2. Write a python program to implement linear SVM.Datset-Iris1.csv
```python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm, datasets
iris = pd.read_csv('Iris1.csv')
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
C = 1.0
svc = svm.SVC(kernel ='linear', C = 1).fit(X, y)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap = plt.cm.Paired, alpha = 0.8)
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()
```


3. Create an HTML form that contains the Employee Registration details and write a JavaScript to validate DOB, Joining Date, and Salary
employee.html
```html

<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>
      JavaScript Form Validation using a sample Employee registration form
    </title>

    <link rel="stylesheet" href="employee.css" type="text/css" />
    <script src="employee.js"></script>
  </head>
  <body onload="document.registration.userid.focus();" bgcolor="orange">
    <h1>Employee Registration Form</h1>
    <form name="registration" onSubmit="return formValidation();">
      <ul>
        <li><label for="first">First Name:</label></li>
        <li><input type="text" name="first" size="50" /></li>
        <li><label for="last">Last Name:</label></li>
        <li><input type="text" name="last" size="50" /></li>
        <li><label for="empid">Employee id:</label></li>
        <li><input type="text" name="empid" size="50" /></li>
        <li><label for="birth">Birth of date:</label></li>
        <li><input type="date" id="birth" name="birth" /></li>
        <li><label for="address">Address:</label></li>
        <li><input type="text" name="address" size="50" /></li>
        <li><label for="country">Country:</label></li>
        <li>
          <select name="country">
            <option selected="" value="Default">
              (Please select a country)
            </option>
            <option value="AF">Australia</option>
            <option value="AL">Canada</option>
            <option value="DZ">India</option>
            <option value="AS">Russia</option>
            <option value="AD">USA</option>
          </select>
        </li>
        <li><label for="no">Contact no:</label></li>
        <li><input type="number" id="" name="no" /></li>
        <li><label for="jdate">Date of joining:</label></li>
        <li><input type="date" id="" name="jdate" /></li>
        <li><label for="email">Email:</label></li>
        <li><input type="text" name="email" size="50" /></li>
        <li><label id="gender">Gender:</label></li>
        <li>
          <input type="radio" name="male" value="Male" /><span>Male</span>
        </li>
        <li>
          <input type="radio" name="female" value="Female" /><span>Female</span>
        </li>
        <li><label for="salary">salary:</label></li>
        <li><input type="number" id="salary" name="salary" /></li>
        <li><input type="submit" name="submit" value="Submit" /></li>
      </ul>
    </form>
  </body>
</html>

```

```js
//employee.js
function formValidation() {
    var first = document.registration.first;
    var last = document.registration.last;
    var empid = document.registration.empid;
    var birth = document.registration.birth;
    var uadd = document.registration.address;
    var ucountry = document.registration.country;
    var no = document.registration.no;
    var jdate = document.registration.jdate;
    var uemail = document.registration.email;
    var umgen = document.registration.umgen;
    var ufgen = document.registration.ufgen;
    var salary = document.registration.salary;
    if (allLetter(first)) {
        if (allLetter(last)) {
            if (alphanumeric(empid)) {
                if (allb(birth)) {
                    if (alphanumeric(uadd)) {
                        if (countryselect(ucountry)) {
                            if (allnumeric(no)) {
                                if (allnumeric(jdate)) {
                                    if (ValidateEmail(uemail)) {
                                        if (validgendor(umgen, ufgen)) {
                                            if (allnumeric(salary)) {
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return false;
}
function allLetter(first) {
    var letters = /^[A-Za-z]+$/;
    if (first.value.match(letters)) {
        alert('employee name submitted');
        return true;
    }
    else {
        alert('employee name must have alphabet characters only');
        first.focus();
        return false;
    }
}
function allLetter(last) {
    var letters = /^[A-Za-z]+$/;
    if (last.value.match(letters)) {
        alert("employee name submitted");
        return true;
    }
    else {
        alert('employee name must have alphabet characters only');
        last.focus();
        return false;
    }
}
function alphanumeric(empid) {
    var letters = /^[0-9a-zA-Z]+$/;
    if (empid.value.match(letters)) {
        alert("employee id submitted");
        return true;
    }
    else {
        alert('employee id must have alphanumeric characters only');
        uadd.focus();
        return false;
    }
}
function allb(birth) {
    var birth_len = birth.value.length;
    if (birth_len == 0) {
        alert("birth date should not be empty");
        birth.focus();
        return false;
    }
    alert("birth of date submitted");
    return true;
}
function alphanumeric(uadd) {
    var letters = /^[0-9a-zA-Z]+$/;
    if (uadd.value.match(letters)) {
        alert("address submitted");
        return true;
    }
    else {
        alert('address must have alphanumeric characters only');
        uadd.focus();
        return false;
    }
}
function countryselect(ucountry) {
    if (ucountry.value == "Default") {
        alert('Select your country from the list');
        ucountry.focus();
        return false;
    }
    else {
        alert("country submitted");
        return true;
    }
}
function allnumeric(no) {
    var number = /^[0-9]+$/;
    if (no.value.match(number)) {
        alert("Contact Number submitted");
        return true;
    }
    else {
        alert('Contact no must have numeric numbers only');
        no.focus();
        return false;
    }
}
function allnumeric(jdate) {
    var jdate_len = jdate.value.length;
    if (jdate_len == 0) {
        alert("date of joining should not be empty");
        birthday.focus();
        return false;
    }
    alert("date of joining submitted");
    return true;
}
function ValidateEmail(uemail) {
    var mailformat = /^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$/;
    if (uemail.value.match(mailformat)) {
        alert("email address is submitted");
        return true;
    }
    else {
        alert("You have entered an invalid email address!");
        uemail.focus();
        return false;
    }
}
function validgender(umgen, ufgen) {
    x = 0;
    if (umgen.checked) {
        x++;
    } if (ufgen.checked) {
        x++;
    }
    if (x == 0) {
        alert('Select Male/Female');
        umgen.focus();
        return false;
    }
    else {
        window.location.reload()
        return true;
    }
}
function allnumeric(salary) {
    var sal = /^[0-9]+$/;
    if (salary.value.match(sal)) {
        alert("salary submitted");
        return true;
    }
    else {
        alert('salry is not submitted');
        salary.focus();
        return false;
    }
}

```

employee.css

```css
h1 {
  margin-left: 70px;
}
form li {
  list-style: none;
  margin-bottom: 5px;
}

form ul li label {
  float: left;
  clear: left;
  width: 100px;
  text-align: right;
  margin-right: 10px;
  font-family: Verdana, Arial, Helvetica, sans-serif;
  font-size: 14px;
}

form ul li input,
select,
span {
  float: left;
  margin-bottom: 10px;
}

form textarea {
  float: left;
  width: 350px;
  height: 150px;
}

[type="submit"] {
  clear: left;
  margin: 20px 0 0 230px;
  font-size: 18px;
}

p {
  margin-left: 70px;
  font-weight: bold;
}

```
