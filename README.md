<div id="top"></div>

# Modeling Survival For The Titanic

  
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
        <li><a href="#how-to-use">How to Use</a> </li>
    <li><a href="#roadmap">Roadmap</a></li>
  </ol>
</details>




## About The Project

This project was created to demonstrate different types of models. This includes

1) Grid Testing
2) Logistic Regression
3) Kfold Testing
4) SVC Testing
5) Decision Tree

The project will display the accuracy, or average correctness, as well as some relevent information for each model in the console.




<p align="right">(<a href="#top">back to top</a>)</p>


---
### Built With

* [Spyder version 4.2.5](https://www.spyder-ide.org)
* [Python version 3.8](https://www.spyder-ide.org)

<p align="right">(<a href="#top">back to top</a>)</p>




<!-- GETTING STARTED -->
## Getting Started


To view the dashboard in its entirety you will need to download the following. 

1) Titanic folder
2) Titanic_modeling_one.py


or clone the repository 
```sh
git clone https://github.com/futious/Titanic_modeling.git
```


  ---
### Installation
 

   ```sh
  pip install panda==1.2.4
   ```


  

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- How to Use -->
## How to Use

Once you run the code the relavent information will populate in the concole.
 <p>
  <img width="1000" align='left' src="https://user-images.githubusercontent.com/49052260/150414521-05966ae7-4360-45c9-9d1b-a46cf11ecba1.png?raw=true">
</p>

This information can be broken down into three parts. 
* The accuracy for the test its self.
* The accuracy using kfold testing.
* The array of outputs when using kfold testing. 

The current setup uses a 3 fold testing system, but that can be changed.

The last 4 rows gives you infomration when you are testing for optimal parameter. 
*The first line shows the total amount of different combinations of parameters. the
*The second line shows the accuracy of the test.
*The third line shws the average of all the different models using the different combinations of parameters.
*The lst line shows the optimal parameters for the model.





<!-- ROADMAP -->
## Roadmap

This project is not yet complete and requires the following.
- [ ] Modifications to the information that the model uses.
- [ ] Multiple iterations of the models.
- [ ] Check the Decision tree model with its 93% accuracy to determine if the model is overfitted.
- [ ] Run the model that is determined to be optimal on the test data set.


<p align="right">(<a href="#top">back to top</a>)</p>
