# PROVA AGENT BASE SIMULATION

population = 10000
contact_per_day = 0.2 
infection_duration = 14 # days

class Person:
    def __init__(self, contact_per_day, infection_duration):
        self.contact_per_day = contact_per_day
        self.infection_duration = infection_duration
        self.disease = False # wheter or not he has the disease
        self.disease_counter = 0 # days of infection

    def set_disease(self, value):
        self.disease = value # must be true or false

    def get_disease(self):
        return self.disease

    def increment_counter(self):
        self.disease_counter += 1

    def get_inf_counter(self):
        return self.disease_counter

    def remaining_infection_days(self):
        return self.infection_duration - self.disease_counter

# start with a person with the disease
first_infected = Person(contact_per_day, infection_duration)
first_infected.set_disease(True) # set the disease

