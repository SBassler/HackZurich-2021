Delage

The app Delage is a platform unifying the Coursera of sports coaches with the Amazon of Digital Health apps to take a holistic view of your health data in order to rejuvenate your biological age.






What if we told you that time travel was possible?

Our app Delage allows you to travel back in time, by taking a holistic view of your health data in order to rejuvenate your biological age. While other solutions stop at this point, Delage takes an active role as your personal health coach and gives you personalised lifestyle suggestions, so you can live a happier and more fulfilling life.

The foundation for Delage is data, which is collected by the supercomputer in your pocket, your smartphone. As with the existing Helsana solution, we welcome data from various sources and providers, and we want to make Delage the one-stop solution where all your precious data can be safely stored. But storing is just the first step, simplifying the jungle of 100 different apps with different login data. 

The second step is the analysis, where modern machine learning technology is utilised, in order to get the most out of your valuable data. Your personal data holds the key to unlock your full health potential, we help you to unlock it.  

While other solutions stop here, Delage goes one step further. Instead of the passive data collector, Delage becomes your active personalised Health Coach, something desirable for many but affordable for few. This is how we want to democratise health and relieve pressure from the health care systems. Delage will integrate all available information and guide your decisions with additional information supplied in a personalised manner. So it will not just tell you to do sports or eat healthily, but also when it matters most and how you should approach it. 

This will help you to unlock your full potential and turn back your biological ageing clock. Here comes another important feature of Delage. Based on existing data, we are working on machine learning models to accurately predict your biological age, as well as how you can rejuvenate it.

We believe that information and guided decisions increase your health span, the time until the onset of age-related diseases dramatically, and thereby take the burden from the health care systems.  Age-related diseases like Diabetes, cardiovascular diseases, cancer or neurodegenerative diseases accumulate the highest monetary cost for health insurance and the highest cost in terms of lost years of lifespan for patients.

How did we implement our prototype?

We first accessed the database directly via Python to download the dataset and make it available for further processing, that which performed via custom-made R and Python scripts.

The key steps of the analysis included the build-up of a tidy dataset with all users and their time-resolved health activity over several years, sorted into the main categories "move" and "recharge". In addition, we imputed a dataset of "eating" information for each user, since this represents a valuable addition for accurately predicting the biological age.

This information was utilised to create personalised visualisations for each user regarding their general performance, as well as their specific performance in the areas "eat", "move" and "recharge". Finally, a summary score for each area (ranging from 0-100), as well as a total summary score (ranging from 0-100), were computed relative to the whole dataset.

We then worked on the problem of predicting the biological age, which unlike the chronological age it is influenced by the lifestyle. The estimate of a single number for the biological age is not enough: the health records can contain a high amount of noise (for example a user could not be wearing a device all the time), so one needs to complement the predictions with the quantification of the uncertainty. A way to address this technically challenging task is by employing hierarchical Bayesian networks to learn the noise parameters associated with the age estimates. To train these model we harness the power of amortized variational inference, made possible by the deep learning framework PyTorch. Given more time and resources, we could complete and refine the model, and integrate the rest of the "eat" and "move" data, to maximize the statistical power of the predictions.

Besides the scientific computing on dataset in the backend, an App prototype was build in Figma and supplied with data and graphs via Bravo.

---

We used the data provided by Zurich Insurance, analysed it and created valuable insights for each unique user. We then went on to visualise the data, creating custom charts for each user and connected these to our prototype via an API. 
