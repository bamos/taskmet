This is the code for the paper "TaskMet: Metric Learning for Model Learning". 

# Todo

- [x] TaskMet class object, -> TaskMet should work as wrapper on existing prediction model with taskloss. So TaskMet can be considered as taking data, prediction model (architecture), task loss as inputs.
- [ ] code for running lodl experiments 
- [ ] code for running omd experiments

# Changes needed in LODL
* fixing relative import, when LODL is used in thirdparty
* have to create models folder inside LODL