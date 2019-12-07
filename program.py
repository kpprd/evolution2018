#!/usr/local/bin/python3
import re
import numpy
import itertools
import scipy.misc
import copy
from math import cos, sin, exp, pi, factorial
import sys
import time

'''
This file is an online appendix to my master thesis 'Chiasma interference, reinforcement, and the evolution of chromosomal inversions', 
which can be found at https://www.duo.uio.no. The program it incodes is a general-purpose deterministic multilocus evolution simulator,
which automatically generates and iterates appropriate recurrence equations based on the user's input. I plan to publish an
extended version with a full user manual sometime in the future.

An example input file is included as an additional online appendix. To run the simulation in the input file 'example.txt', set the 
working directory to the appropriate folder and execute the following commands in a Python 3 environment:

from program import Simulation
simulation = Simulation()
simulation.read_input('example.txt')
simulation.run()

Enjoy your general-purpose deterministic multilocus evolution experience!

Oystein Kapperud, December 2018
'''


class Population:
	'''
	A class for storing and manipulating information relating to the full population.
	
	Key attributes:
	self.demes/chromosomes/males/females/couples: A list of instances of class Deme/Chromosome/Male/Female/Couple. Each instance can be 
		identified by its unique index in the list.
	self.genotypes: a list of instances of class Genotype, males and females (note that classes Male and Female are subclasses of class Genotype)
	self.demes_n/females_n/males_n: the number of demes/female genotypes/male genotypes
	self.male_fitness_array/female_fitness_array: An array giving the fitness of each male/female in each deme.
	self.female_offspring_tensor/male_offspring_tensor: A matrix in which element i,j gives the proportion of
		female/male j among female/male offspring of couple i.
	self.error1/error2: (float) The accepted error when estimating infinite series expressions for homokaryotypes/heterokaryotypes.
		The results for lower values are more accurate, but take longer to compute.
	self.g_values: The values of the function g as defined in the thesis
	self.b_values: The values of the function b as defined in the thesis
	self.pi_vector: The stationary distribution
	self.gamma_values: The intervening O'' event distribution.
	self.male_frequencies/female_frequencies: an array giving the initial frequencies of each male/female in each deme
	self.male_offspring_tensor/female_offspring_tensor: Numpy array used in simulations. Element i,j gives the proportion of male/female with index j (male.index = j) among 
		all male/female offspring of couple with index i (couple.index = i)
	self.preference_tensor: Numpy array used in simulations. Element i,j,k gives the probability that female with index k will accept male with index j after a single encounter
		in deme i.
	self.male_fitness_array/female_fitness_array: A numpy array with the fitness of each male/female in each deme (sorted by index)
	self.alpha: (bool) The alpha parameter as defined in the thesis. True if there is interfernce across breakpoint boundaries
	self.male_recombinant: (bool) True if there is recombination in males
	self.c: (float) The c parameter as defined in the thesis; In short, the cost of a single search.
	self.male_key/female_key: (str) The locus diplotype key for the sex determination locus. These can either be '$01' (heterogametic sex) or '$11' (homogametic sex).
	
	'''
	def __init__(self, simulation):
		self.chromosomes = []
		self.sex_chromosome = None
		self.sex_chromosome_index = None
		self.males = []
		self.females = []
		self.male_keys = []
		self.female_keys = []
		self.couples = []
		self.couple_keys = []
		self.haplotypes = []
		self.diplotypes_alleles = []
		self.chromosome_diplotypes = []
		self.demes = []
		self.deme_keys = []
		self.alpha = True # default
		self.simulation = simulation
		self.b_values = [1.0]
		self.heterokaryotype_truncate = []
		self.default_migration = None
		self.fitness_interaction = None
		self.preference_interaction = None
		self.fitness_contribution_general_keys = []
		self.preference_contribution_general_keys = []
		self.linear_meiosis = False
		self.g_values = None
		self.max_truncate = None
		self.pi_vector = None
		self.male_fitness_array = []
		self.female_fitness_array = []
		self.loci = {}
		self.alleles = []
		self.alleles_dictionary = {}
		self.haplotypes = []
		self.continent = None
		self.dynamic_deme_indices = []
		self.static_deme_indices = []
		self.male_recombinant = True # default
		self.haplotypes_to_remove = []
		self.c = 0.0 # default
		self.save_offspring_filename = None
		self.load_offspring_filename = None
		self.error1 = 1.0e-14 # default
		self.error2 = 1.0e-12 # default
		self.closed_form = False # default
	
	
	def print_offspring_frequencies(self):
		print('Printing offspring frequencies...')
		for c in range(self.couples_n):
			print('Couple:', self.couples[c].key)
			print('Female offspring:')
			for f in range(self.females_n):
				print(self.female_keys[f], self.female_offspring_tensor[c][f])
			print('\nMale offspring:')
			for m in range(self.males_n):
				print(self.male_keys[m], self.male_offspring_tensor[c][m])
			print('')
	
	def print_gamete_frequencies(self):
		print('Printing gamete frequencies...')
		for genotype in self.genotypes:
			print(genotype.key)
			print(genotype.gametes)
			print('Sum:', sum(genotype.gametes.values()))
			print('')
	
	def initialize_offspring_tensors(self):
		self.female_offspring_tensor = numpy.zeros((self.couples_n, self.females_n))
		self.male_offspring_tensor = numpy.zeros((self.couples_n, self.males_n))
	
	def normalize_offspring_tensors(self):
		self.female_offspring_tensor = (self.female_offspring_tensor.transpose()/numpy.sum(self.female_offspring_tensor, axis = 1)).transpose()
		self.male_offspring_tensor = (self.male_offspring_tensor.transpose()/numpy.sum(self.male_offspring_tensor, axis = 1)).transpose()
	
	def initialize_preference_tensor(self):
		self.preference_tensor = numpy.ones((self.demes_n, self.males_n, self.females_n))  
	
	def find_offspring(self):
		print('Finding offspring...')
		for couple in self.couples:
			couple.find_offspring()

	
	def calculate_gamete_frequencies(self):
		print('Calculating gamete frequencies...')
		for c in self.chromosome_diplotypes:
			for chromosome_diplotype in c:
				chromosome_diplotype.calculate_gamete_frequencies()
		
		for genotype in self.genotypes:
			genotype.calculate_gamete_frequencies()
	
	
	def print_karyotypes(self):
		print('Karyotypes')
		for chromosome in self.chromosomes:
			print(chromosome.key)
			for diplotype in chromosome.diplotypes:
				print(diplotype.key)
				for karyotype in diplotype.karyotypes:
					print(karyotype.key)
					print('')
	
	def generate_pi_vector(self):
		'''
		Generates the stationary distribution and stores it as self.pi_vector
		'''
		pi_vector = numpy.zeros(self.m+1, numpy.float64)
		denominator = 0.0
		gamma_values = self.gamma_values
		for q in range(self.m+1):
			denominator += (q+1)*gamma_values[q]
		for i in range(self.m+1):
			nominator = 0.0
			for q in range(i, self.m+1):
				nominator += gamma_values[q]
			pi_vector[i] = nominator/denominator
		
		self.pi_vector = pi_vector
	

	
	def generate_g_values(self, maximum):
		'''
		Initializes the g values matrix
		'''
		g_values = numpy.zeros((maximum,maximum), numpy.float64)
		g_values[0][0] = 1.0
		for n in range(1,maximum):
			for s in range(1,maximum):
				if n<s:
					break
				t = 0
				for k in range(s-1, n):
					if n-1-k<len(self.gamma_values):
						t+=g_values[k][s-1]*self.gamma_values[n-1-k]
				g_values[n][s] = t
				
		self.g_values = g_values
		
	
	def extend_g_values(self, extend_by):
		'''
		Extends the g values matrix if needed
		'''
		old_size = self.g_values.shape[0]
		new_size = old_size+extend_by
		new_g_values = numpy.zeros((new_size, new_size))
		new_g_values[0:old_size, 0:old_size] = self.g_values
		for n in range(old_size,new_size):
			for s in range(1,new_size):
				if n<s:
					break
				t = 0
				for k in range(s-1, n):
					if n-1-k<len(self.gamma_values):
						t+=new_g_values[k][s-1]*self.gamma_values[n-1-k]
				new_g_values[n][s] = t
		self.g_values = new_g_values
			
	
	def print_fitness(self):
		print('Printing fitness...')
		for genotype in self.genotypes:
			print(genotype.key)
			print(genotype.fitness)
			print('')
	
	def calculate_fitness(self):
		print('Calculating fitness...')
		if self.fitness_interaction == None:
			print('OBS! No fitness interaction found. Implementing multiplicative fitness interaction by default')
		for male in self.males:
			male.calculate_fitness()
		for female in self.females:
			female.calculate_fitness()
		
		self.male_fitness_array = numpy.array(self.male_fitness_array).transpose()  # [deme][male]
		self.female_fitness_array = numpy.array(self.female_fitness_array).transpose() # [deme][female]
		

	
	def calculate_preferences(self):
		print('Calculating preferences...')
		if self.preference_interaction == None:
			print('OBS! No preference interaction found. Implementing multiplicative preference interaction by default')
		for couple in self.couples:
			couple.calculate_preference()
			
	def print_preference_contributions(self):
		print('Printing preference contributions...')
		for couple in self.couples:
			print(couple.key)
			print(couple.preference_contributions)
			print('')
	
	def print_preferences(self):
		print('Printing preferences...')
		for couple in self.couples:
			print(couple.key)
			print(couple.preference)
			print('')
	
	def initialize_couples(self):
		index = 0
		for male in self.males:
			for female in self.females:
				couple = Couple(self, female, male, index)
				self.couples.append(couple)
				self.couple_keys.append(couple.key)
				index += 1
		self.couples_n = len(self.couples)
	
	def set_migration_matrix(self, key, matrix): 
		if key == 'default':
			self.migration_matrix = matrix
	
	def print_fitness_contributions(self):
		print('Printing fitness contributions:')
		for genotype in self.genotypes:
			print(genotype.key)
			print(genotype.fitness_contributions)
			print('')
		
	
	def initialize_fitness_contributions(self):
		print('Initializing fitness contributions...')
		for deme in self.demes:
			deme.initialize_fitness_contributions()
	
	def initialize_preference_contributions(self):
		print('Initializing preference contributions...')
		for deme in self.demes:
			deme.initialize_preference_contributions()
	
	def set_sex_keys(self):
		# $ is the sex locus, so $01 and $11 indicate heterozygosity and homozygosity, respectively, at this locus
		if self.male_heterogametic:
			self.male_key = '$01'
			self.female_key = '$11'
		else:
			self.male_key = '$11'
			self.female_key = '$01'
		
	def make_genotype_array(self):
		self.genotypes = self.females + self.males
		self.genotype_keys = self.female_keys + self.male_keys
	
	def set_genotype_frequencies(self, new_male_genotype_frequencies, new_female_genotype_frequencies):
		'''
		The initial male/female frequencies are automatically assumed to be in Hardy-Weinberg equilibrium if no additional
		information is given. This function allows the user to change the genotype frequencies to any values.
		
		Arguments:
		new_male_genotype_frequencies: a list of dictionaries formatted as [{male_key: new_male_frequency}], where the index of the list
			correspond to the index of the deme.
		new_female_genotype_frequencies: same, but for females
		'''
		male_change = [[] for d in range(self.demes_n)]
		male_not_change = [[] for d in range(self.demes_n)]
		female_change = [[] for d in range(self.demes_n)]
		female_not_change = [[] for d in range(self.demes_n)]
		male_change_old_sums = []
		female_change_old_sums = []
		male_change_new_sums = []
		female_change_new_sums = []
		for d in range(self.demes_n):
			for m in range(self.males_n):
				if self.male_keys[m] in new_male_genotype_frequencies[d].keys():
					male_change[d].append(m)
				else:
					male_not_change[d].append(m)
			for f in range(self.females_n):
				if self.female_keys[f] in new_female_genotype_frequencies[d].keys():
					female_change[d].append(f)
				else:
					female_not_change[d].append(f)
					
			# male_change/female_change is now a list of lists of indices of male/female genotypes whose frequencies shall be changed.
			# male_change[d] gives the index of male genotypes in deme with index d
			
			male_change_old_sums.append(numpy.sum(self.male_frequencies[d][male_change[d]])) # the sum of the old frequncies of the males whose frequencies are to be changed
			female_change_old_sums.append(numpy.sum(self.female_frequencies[d][female_change[d]]))
			male_change_new_sums.append(sum(new_male_genotype_frequencies[d].values())) # the sum of the new frequncies of the males whose frequencies are to be changed
			female_change_new_sums.append(sum(new_female_genotype_frequencies[d].values()))
		
		# update self.male_frequencies and self.female_frequencies:
		for d in range(self.demes_n):
			for m in range(self.males_n):
				male_key = self.male_keys[m]
				if male_key in new_male_genotype_frequencies[d]:
					self.male_frequencies[d][m] = new_male_genotype_frequencies[d][male_key]
				else:
					if 1-male_change_old_sums[d] != 0:
						self.male_frequencies[d][m] = self.male_frequencies[d][m]*((1-male_change_new_sums[d])/(1-male_change_old_sums[d]))
					else:
						self.male_frequencies[d][m] = (1.0-male_change_new_sums[d])/len(male_not_change[d])
			for f in range(self.females_n):
				female_key = self.female_keys[f]
				if female_key in new_female_genotype_frequencies[d]:
					self.female_frequencies[d][f] = new_female_genotype_frequencies[d][female_key]
				else:
					if 1-female_change_old_sums[d] != 0:
						self.female_frequencies[d][f] = self.female_frequencies[d][f]*((1-female_change_new_sums[d])/(1-female_change_old_sums[d]))
					else:
						self.female_frequencies[d][f] = (1.0-female_change_new_sums[d])/len(female_not_change[d])
	
	def initialize_frequency_arrays(self):
		self.male_frequencies = [[] for i in range(len(self.demes))]
		self.female_frequencies = [[] for i in range(len(self.demes))]
		for m in range(len(self.males)):
			male = self.males[m]
			male.set_index(m) # store the index as an attribute of the male instance
			for d in range(self.demes_n):
				self.male_frequencies[d].append(male.initial_frequencies[d]) # the initial_frequencies[d] values are the Hardy-Weinberg proportions
		
		for f in range(len(self.females)):
			female = self.females[f]
			female.set_index(f)
			for d in range(self.demes_n):
				self.female_frequencies[d].append(female.initial_frequencies[d])
		self.male_frequencies = numpy.array(self.male_frequencies)
		self.female_frequencies = numpy.array(self.female_frequencies)
		
		# normalize frequencies
		for d in range(len(self.demes)):
			sf = numpy.sum(self.female_frequencies[d])
			sm = numpy.sum(self.male_frequencies[d])
			if sm != 0:
				self.male_frequencies[d] = self.male_frequencies[d]/sm
			if sf != 0:
				self.female_frequencies[d] = self.female_frequencies[d]/sf
		
	
	def calculate_all_recombination_pattern_probabilitites(self):
		print('Calculates recombination pattern probabilitites...')
		for chromosome in self.chromosomes:
			chromosome.calculate_recombination_pattern_probabilities()
	
	def extend_b_values(self, extend_by = 10):
		'''
		Initialize or extend the array of b values.
		'''
		gamma_values = self.gamma_values
		b_values = self.b_values
		m = self.m
		current_length = len(b_values)
		for n in range(current_length, current_length + extend_by):
			s = 0
			if n-1>=m:
				for q in range(n-1-m, n):
					s += b_values[q]*gamma_values[n-1-q]
			else:
				for q in range(n):
					s += b_values[q]*gamma_values[n-1-q]
			b_values.append(s)
		self.b_values = b_values
	
	def add_calculator(self, calculator):
		self.calculator = calculator
	
	def set_gamma_values(self, gamma_values):
		'''
		Removes trailing zeros from gamma_values.
		'''
		while gamma_values[-1] == 0:
			gamma_values = gamma_values[:-1]
		self.gamma_values = gamma_values
		
		self.m = len(self.gamma_values) -1
				
	
	def print_initial_genotype_frequencies(self):
		print('Printing initial genotype frequencies...')
		print('Males:')
		male_frequencies = self.male_frequencies.transpose()
		for m in range(self.males_n):
			print(self.males[m].key)
			print(male_frequencies[m])
			print('')
		print('Females:')
		female_frequencies = self.female_frequencies.transpose()
		for f in range(self.females_n):
			print(self.females[f].key)
			print(female_frequencies[f])
			print('')
		
	def set_demes_n(self):
		self.demes_n = len(self.demes)
	
	def print_genotypes(self):
		self.print_males()
		self.print_females()
	
	def print_males(self):
		print('Males:')
		for m in self.males:
			print(m.key)
	
	def print_females(self):
		print('Females:')
		for f in self.females:
			print(f.key)
	
	def initialize_diplotypes(self):
		for c in self.chromosomes:
			c.initialize_haplotypes()
			c.initialize_diplotypes()
			self.chromosome_diplotypes.append(c.diplotypes)
		
	
	def add_chromosome(self, chromosome):
		for locus in chromosome.loci:
			self.loci[locus.key] = locus
		self.chromosomes.append(chromosome)
	
	def print_chromosomes(self):
		for chromosome in self.chromosomes:
			print(chromosome.loci_keys)
		
	
	def initialize_genotypes(self):
		self.sexlocus_diplo_index = None # the index of the sex locus in the full diplotype key
		for diplotype_set in itertools.product(*self.chromosome_diplotypes): # Loop over all combinations of chromosome diplotypes. Each such combination is a full genotype
			male_bool = diplotype_set[self.sex_chromosome_index].male # male_bool is a boolean variable that is true if the genotype is male
			initial_frequencies = [1.0 for i in range(self.demes_n)]
			for d in diplotype_set:
				for i in range(len(initial_frequencies)):
					initial_frequencies[i] *= d.initial_frequencies[i] # Hardy-Weinberg
			if male_bool:
				male = Male(diplotype_set, self, male_bool, initial_frequencies) # new Male instance
				male.make_key()
				self.males.append(male)
				if self.sexlocus_diplo_index is None:
					self.sexlocus_diplo_index = male.key.index('$')
				# remove sex locus keys from the genotype key, as this information is redundant (when the sex is known)
				if self.male_heterogametic:
					self.male_keys.append(male.key.replace('$01', '').replace('$10', ''))
				else:
					self.male_keys.append(male.key.replace('$11',''))
			else:
				female = Female(diplotype_set, self, male_bool, initial_frequencies)
				female.make_key()
				self.females.append(female)
				if self.male_heterogametic:
					self.female_keys.append(female.key.replace('$11', ''))
				else:
					self.female_keys.append(female.key.replace('$01','').replace('$10',''))
					
		self.males_n = len(self.males)
		self.females_n = len(self.females)
		self.couples_n = self.females_n*self.males_n
		
		self.female_zeros = numpy.zeros((self.demes_n, self.females_n))
		self.male_zeros = numpy.zeros((self.demes_n, self.males_n))


class Genotype:
	'''
	A class for storing and manipulating information relating to a single genotype.
	
	Key attributes:
	self.chromosome_diplotypes: The tuple of chromosome diplotypes making up the genotype.
	self.male: (boolean) True if the genotype is male, False if the genotype is female.
	self.chromosome_diplotypes: The set of chromosome diplotypes making up the full genotype
	self.initial_frequencies: An array with the initial freqency of the genotype in each deme
	self.fitness_contributions: A list of dictionaries with fitness contributions for each deme. E.g. self.fitness_contributions = [{'A': 1.1, 'B&C': 0.8}, {'A': 1.0, 'B&C': 1.0}] indicate that the genoytpe gets a contribution 1.1 from
		locus A and a contribution 0.8 from the interaction between loci B and C in deme 0, and a contribution of 1.0 from both in deme 1. The interaction between the contribution from A and the contribution from B&C is determined
		by the Population attribute fitness_interaction. E.g. fitness_interaction = 'A*B&C' imply multiplicative interaction.
	self.fitness: (list) The fitness of the genotype in each deme
	self.gametes: (dict) Keys are tuples of strings representing the haplotype of a gamete, where each string give the allele keys (sorted by the order of the loci) of the loci on a single chromosome. The elements are sorted by the order
		the chromosomes are given in the input file. Values are the proportions of each gamete among all the gametes produced by the genotype.
	self.key: (str) An identifier made up of the keys to each individual chromosome diplotype, concatenated, in the order the chromosomes appear in the input file.
	self.index: (int) An integer used to identify the male/female among the other males/females (the index is unique among males/females, but one male and one female
		 can have the same index). Also used to store and retrieve information at appropriate indices in arrays and lists.
	
	
	
	'''
	def __init__(self, chromosome_diplotypes, population, male, initial_frequencies):
		self.chromosome_diplotypes = chromosome_diplotypes
		self.population = population
		self.male = male
		self.initial_frequencies = initial_frequencies
		locus_diplotypes = []
		self.locus_allele_keys = []
		for chromosome_diplotype in chromosome_diplotypes:
			loci_keys = chromosome_diplotype.chromosome.loci_keys
			for i in range(len(loci_keys)):
				allele_keys = sorted([str(chromosome_diplotype.haplotypes[0].allele_keys[i]), str(chromosome_diplotype.haplotypes[1].allele_keys[i])])
				locus_diplotypes.append(loci_keys[i] + allele_keys[0] + allele_keys[1])
				self.locus_allele_keys.append(loci_keys[i] + allele_keys[0])
				self.locus_allele_keys.append(loci_keys[i] + allele_keys[1])
		self.locus_diplotypes = locus_diplotypes
		self.fitness_contributions = None
		self.fitness = None
	
	def calculate_gamete_frequencies(self):
		'''
		Calculates and stores the gamete frequencies of the full genotype. The frequency of a given multi-chromosome gamete is simply the frequencies of each
		single-chromosome haplotype (among all haplotypes produced by the corresponding chromosome diplotype) multiplied.
		'''
		self.gametes = {}
		chromosome_gamete_dictionaries = []
		if self.male and not self.population.male_recombinant:
			for chromosome_diplotype in self.chromosome_diplotypes:
				if chromosome_diplotype.nonrecombinant_gametes is None:
					chromosome_diplotype.generate_nonrecombinant_gametes()
				chromosome_gamete_dictionaries.append(chromosome_diplotype.nonrecombinant_gametes)
		else:
			for chromosome_diplotype in self.chromosome_diplotypes:
				chromosome_gamete_dictionaries.append(chromosome_diplotype.gametes)

		for gamete_key in itertools.product(*chromosome_gamete_dictionaries): # cartesian product
			frequency = 1.0
			for i in range(len(chromosome_gamete_dictionaries)):
				frequency *= chromosome_gamete_dictionaries[i][gamete_key[i]]
			self.gametes[gamete_key] = frequency
	
	def calculate_fitness(self):
		'''
		Calculates and stores the fitness of the genotype in each deme. See self.fitness_contributions in the class Genotype docstring.
		'''

		if self.fitness is None:
			self.fitness = [[] for i in range(self.population.demes_n)]
		if self.fitness_contributions is None:
			self.fitness_contributions = [[] for i in range(self.population.demes_n)]
			self.fitness = [1.0 for i in range(self.population.demes_n)]
			
		for d in range(len(self.fitness_contributions)):
			expression = self.population.fitness_interaction
			if expression is not None:
				for general_key in self.population.fitness_contribution_general_keys: # general_key refers to loci (e.g. 'A' or 'A&B') but not specific alleles.
					if general_key in self.fitness_contributions[d]:
						contribution = str(self.fitness_contributions[d][general_key])
					else:
						contribution = '1.0' # default fitness contribution
					if general_key in expression:
						expression = expression.replace(general_key, contribution)
					else:
						print('OBS! General key', general_key, 'not found in fitness interaction!')
		
				expression_pattern = '(?P<expression>[\d.*/\-+]+)'    # check that the expression only contains numbers and valid operators
				expression_match = re.search(expression_pattern, expression)
				error = False
				if expression_match:
					if expression_match.group('expression') == expression:
						fitness = eval(expression)
						self.fitness[d] = fitness
					else:
						error = True
				else:
					error = True
			
				if error:
					print('Error! Invalid fitness interaction!')
					sys.exit(1)
		
			
			else: # default to multiplicative fitness interaction if no other information is given
				fitness = 1.0
				for general_key in self.fitness_contributions[d]:
					fitness *= self.fitness_contributions[d][general_key]
				self.fitness[d] = fitness
		if self.male:
			self.population.male_fitness_array.append(self.fitness)
		else:
			self.population.female_fitness_array.append(self.fitness)
	
	def make_key(self):
		key = ''
		for d in self.chromosome_diplotypes:
			key += d.key
		self.key = key
		return key
	
	def set_index(self, index):
		self.index = index
		for chromosome_diplotype in self.chromosome_diplotypes:
			chromosome_diplotype.add_index(index, self.male)
	
	def add_fitness_contribution(self, deme_index, general_key, fitness):
		if self.fitness_contributions is None:
			self.fitness_contributions = [{} for i in range(self.population.demes_n)]
		self.fitness_contributions[deme_index][general_key] = fitness
		if not general_key in self.population.fitness_contribution_general_keys:
			self.population.fitness_contribution_general_keys.append(general_key)

class Female(Genotype):
	'''
	A subclass of class Genotype. The only additional attribute is self.preference_contributions (see class Couple)
	'''
	def __init__(self, haplotypes, population, male, initial_frequencies):
		Genotype.__init__(self, haplotypes, population, male, initial_frequencies)
		self.preference_contributions = {}


class Male(Genotype):
	'''
	A subclass of class Genotype.
	'''
	def __init__(self, haplotypes, population, male, initial_frequencies):
		Genotype.__init__(self, haplotypes, population, male, initial_frequencies)
		
class Couple:
	'''
	A class for storing and manipulating information relating to a single couple (one male and one female)
	
	Key attributes:
	self.male: Pointer to the Male instance
	self.female: Pointer to the Female instance
	self.index: (int) A unique integer. Used to identify the couple and to store and retrieve information at appropriate indices in arrays and lists.
	self.key: (str) A concatenation of the female key, an 'x' character, and the male key.
	self.preference_contributions: A list of dictionaries giving the contribution of each pair of preference/trait alleles to the probability that the female
		will accept the male at a single encounter, for each deme. E.g. self.preference_contributions = [{'PxT': 0.9, 'AxB': 0.8}, {'PxT':1.0, 'AxB': 1.0}] means
		that the couple get a contribution of 0.9 from preference locus P and trait locus T and a contribution 0.8 from preference locus A and trait locus B, in deme 0,
		and a contribution 1.0 from both pairs in deme 1. The probability that the female will accept the male at a given encounter is then calculated according to the
		Population attribute preference_interaction. E.g. preference_interaction = 'PxT*AxB' indicate multiplicative interaction.
	
		
	
	'''
	def __init__(self, population, female, male, index):
		self.male = male
		self.female = female
		self.index = index
		self.key = female.key + 'x' + male.key
		self.preference_contributions = None
		self.population = population
		self.offspring = {}
		self.preference = None
		self.preference_contributions = None
		self.f = female.index
		self.m = male.index

	def calculate_preference(self):
		'''
		Calculates the probability that the female will accept the male on a single encounter, for each deme. See self.preference_contributions in the docstring for class Couple.
		'''
		if self.preference is None:
			self.preference = [[] for i in range(self.population.demes_n)]
		if self.preference_contributions is None:
			self.preference_contributions = []
		expression = self.population.preference_interaction
		for d in range(len(self.preference_contributions)):
			if expression is not None:
				for general_key in self.population.preference_contribution_general_keys:
					if general_key in self.preference_contributions[d]:
						contribution = str(self.preference_contributions[d][general_key])
					else:
						contribution = '1.0' # default contribution
					if general_key in expression:
						expression = expression.replace(general_key, contribution)
				
				# check that the expression only contains numbers and operators:
				expression_pattern = '(?P<expression>[\d.*/\-+]+)'
				expression_match = re.search(expression_pattern, expression)
				error = False
				if expression_match:
					if expression_match.group('expression') == expression:
						try:
							preference = eval(expression)
							self.preference[d] = preference
						except:
							print('Error! Invalid input for preference or preference interaction!')
							sys.exit(1)
					else:
						error = True
				else:
					error = True
			
				if error:
					print('Error! Invalid input for preference or preference interaction!')
					sys.exit(1)
		
			
			else: # multiplicative interaction by default
				preference = 1.0
				for general_key in self.preference_contributions[d]:
					preference *= self.preference_contributions[d][general_key]
				self.preference[d] = preference
			self.population.preference_tensor[d][self.m][self.f] = preference
	
	def find_offspring(self):
		'''
		Finds the proportion of each possible male and female among the couple's offspring.
		'''
		chromosomes = self.population.chromosomes
		sexlocus_index = self.population.sex_chromosome.sexlocus_index
		for male_gamete, male_gamete_frequency in self.male.gametes.items(): # loop over all gametes generated by the male
			for female_gamete, female_gamete_frequency in self.female.gametes.items(): # loop over all gametes generated by the female
				frequency = male_gamete_frequency*female_gamete_frequency
				if 'unbalanced' in male_gamete or 'unbalanced' in female_gamete:
					if frequency != 0:
						if 'inviable' in self.offspring:
							self.offspring['inviable'] += frequency # unbalanced gametes produce inviable offspring
						else:
							self.offspring['inviable'] = frequency
				else:
					key = ''
					for c in range(len(male_gamete)):
						chromosome = chromosomes[c]
						if chromosome.sex:
							if male_gamete[c][sexlocus_index] < female_gamete[c][sexlocus_index]: # microgametes are by convention sorted first
								chromosome_haplotypes = [male_gamete[c], female_gamete[c]]
							elif female_gamete[c][sexlocus_index] < male_gamete[c][sexlocus_index]:
								chromosome_haplotypes = [female_gamete[c], male_gamete[c]]
							else:
								chromosome_haplotypes = sorted([male_gamete[c], female_gamete[c]])
						else:
							chromosome_haplotypes = sorted([male_gamete[c], female_gamete[c]])
						
						for l in range(len(chromosome.loci_keys)):
							key += chromosome.loci_keys[l]
							key += chromosome_haplotypes[0][l]
							key += chromosome_haplotypes[1][l]
					# key is a string that identifies the offspring
					if key in self.offspring:
						self.offspring[key] += frequency
					else:
						self.offspring[key] = frequency
					if self.population.male_key in key:
						key = key.replace(self.population.male_key, '') # remove the sex identifier ($01/$11) as it is redundant
						self.population.male_offspring_tensor[self.index][self.population.male_keys.index(key)] += frequency*2.0 # store the frequency in the numpy array used in running simulations
					elif self.population.female_key in key:
						key = key.replace(self.population.female_key, '')
						self.population.female_offspring_tensor[self.index][self.population.female_keys.index(key)] += frequency*2.0

		
		if abs(1.0-sum(self.offspring.values())) > 0.001:
			print('Warning! Sum of offspring frequencies for genotype ' +self.key + ' is significantly less than 1. Consider lowering the accepted error (parameters error1 and error2 under primary header #Population) for greater accuracy.')
			print('Sum: ', sum(self.offspring.values()))
			print('')
		
	
	def add_preference_contribution(self, deme_index, general_key, fitness):
		if self.preference_contributions is None:
			self.preference_contributions = [{} for i in range(self.population.demes_n)]
		self.preference_contributions[deme_index][general_key] = fitness
		if not general_key in self.population.preference_contribution_general_keys:
			self.population.preference_contribution_general_keys.append(general_key)

	
class Karyotype:
	'''
	A class for storing and manipulating information relating to a single karyotype. Karyotype here refers to information about the arrangement on a single chromosome.
	The boolean variables self.original and self.homokaryotype define four different karyotypes:
	Original/ancestral homokaryotype: (self.original = True, self.homokaryotype = True) an inversion homokaryotype with loci arranged in the order given in the input file
	Reversed/derived homokaryotype: (self.original = False, self.homokaryotype = True) an inversion homokaryotype with the order of the loci inside the inverted region reversed
		with respect to the order in the input file
	Original heterokaryotype: (self.original = True, self.homokaryotype = False) an inversion heterokaryotype where the interference signal move in the direction given by
		the order of the loci in the input file
	Reversed heterokaryotype: (self.original = False, self.homokaryotype = False) an inversion heterokaryotype where the interference signal move in the direction given when
		the order of the loci inside the inverted region is reversed with respect to the order in the input file. (see figure 2.3 in the thesis).
	
	For heterokaryotypes with interference across the breakpoint boundaries (self.population.alpha = True), the recombination pattern for both the original and the reversed
		heterokaryotype is calculated, and the average of the two is used to calculate the gamete frequencies of the chromosome diplotype.
	
	Other key attributes:
	self.chromosome: Pointer to the Chromosome instance
	self.heteromorphic: (bool) True if there is no recombination on the chromosome
	self.recombination_patterns: A dictionary with the frequencies of the different recombination patterns.
	self.lambda_values/mu_values/d_values/loci/intervals_n etc: see class Chromosome
	
	
	
	'''
	def __init__(self, chromosome, original, homokaryotype, heteromorphic = False):
		self.chromosome = chromosome
		self.original = original
		self.homokaryotype = homokaryotype
		self.heteromorphic = heteromorphic
		if self.heteromorphic:
			self.key = 'heteromorphic'
		else:
			if self.original:
				if self.homokaryotype:
					self.key = 'original_homokaryotype'
				else:
					self.key = 'original_heterokaryotype'
			else:
				if self.homokaryotype:
					self.key = 'reversed_homokaryotype'
				else:
					self.key = 'reversed_heterokaryotype'
		self.population = chromosome.population
		
		if self.heteromorphic:
			self.lambda_values = [0.0 for i in range(len(self.chromosome.lambda_values))]
			self.mu_values = [0.0 for i in range(len(self.chromosome.mu_values))]
			self.d_values = [0.0 for i in range(len(self.chromosome.d_values))]
			self.loci = self.chromosome.loci
		elif self.original:
			self.lambda_values = self.chromosome.lambda_values
			self.mu_values = self.chromosome.mu_values
			self.d_values = self.chromosome.d_values
			self.loci = self.chromosome.loci
		else:
			l = self.chromosome.left_breakpoint_index
			r = self.chromosome.right_breakpoint_index
			lv = self.chromosome.lambda_values
			mv = self.chromosome.mu_values
			dv = self.chromosome.d_values
			self.lambda_values = lv[0:l] + lv[l:r][::-1] + lv[r:]
			self.mu_values = mv[0:l] + mv[l:r][::-1] + mv[r:]
			self.d_values = dv[0:l] + dv[l:r][::-1] + dv[r:]
			
		if self.homokaryotype or self.heteromorphic:
			self.intervals_n = self.chromosome.intervals_n
			self.left_intervals_n = self.intervals_n
			self.inversion_intervals_n = 0
			self.proximal_intervals_n = 0
			self.right_intervals_n = 0
		
		else:
			self.intervals_n = self.chromosome.intervals_n
			self.left_intervals_n = self.chromosome.left_intervals_n
			self.inversion_intervals_n = self.chromosome.inversion_intervals_n
			self.proximal_intervals_n = self.chromosome.proximal_intervals_n
			self.right_intervals_n = self.chromosome.right_intervals_n
			self.lambda_values = list(numpy.array(self.lambda_values)*numpy.array(self.d_values))
			self.mu_values = list(numpy.array(self.mu_values)*numpy.array(self.d_values))
	
	def find_indices(self, pattern, statespace):
		'''
		Returns the indices of states in statespace that match the pattern. The pattern is ordinarily given as a list of boolean variables indicating recombination
		or non-recombination in each interval (first n elements, where n is the number of intervals) and an integer indicating the tetrad configuration (last element).
		Wild-card elements in the pattern are indicated with an 'x' (str).
		See the thesis, chapter 2, for details on the statespace and on tetrad configurations. 
		
		Pattern can also be a string indicating five special cases:
		pattern = 'balanced'/'unbalanced': returns the indices of states with balanced/unbalanced chromatids in the statespace
		pattern = 'balanced_a1'/'unbalanced_a1': returns the indices of states with balanced/unbalanced chromatids and anaphase I tetrad configuration in the statespace
		pattern = 'a1': returns the indices of states with anaphase I tetrad configuration in the statespace
		'''
		left_intervals_n = self.left_intervals_n
		inversion_intervals_n = self.inversion_intervals_n
		match = []
		for i in range(len(statespace)):
			state = statespace[i]
			found = True
			start = left_intervals_n
			end = left_intervals_n + inversion_intervals_n
			if pattern == 'unbalanced':
				if sum(state[start:end])%2 !=0:
					match.append(i)
		
			elif pattern == 'balanced':
				if sum(state[start:end])%2 == 0:
					match.append(i)
		
			elif pattern == 'unbalanced_a1':
				if sum(state[start:end])%2 !=0 and state[-1] == 1:
					match.append(i)
			elif pattern == 'balanced_a1':
				if sum(state[start:end])%2 == 0 and state[-1] == 1:
					match.append(i)
			elif pattern == 'a1':
				if state[-1] == 1:
					match.append(i)
			else:
				for j in range(len(state)):
					if pattern[j] != 'x' and state[j] != pattern[j]:
						found = False
						break
				if found:
					match.append(i)
		return match
	

	
	def calculate_recombination_pattern_probabilities(self):
		'''
		Chooses the appropriate algorithm for calculating recombination pattern probabilities
		'''
		if (self.chromosome.sex and self.heteromorphic): # no recombination in hetermomorphic sex chromosomes.
			self.recombination_patterns = {'unbalanced': 0.0, tuple([False for i in range(self.intervals_n)]): 1.0}
		elif self.chromosome.paracentric and self.population.linear_meiosis and not self.homokaryotype:
			self.calculate_recombination_pattern_probabilities_1()
		else:
			self.calculate_recombination_pattern_probabilities_2()
	
	def calculate_recombination_pattern_probabilities_1(self):
		'''
		Implements theorem 7 in the thesis.
		'''
		calculator = self.chromosome.population.calculator
		poisson  = calculator.poisson
		comb = scipy.misc.comb
		m = self.chromosome.population.m
		lambda_values = self.lambda_values
		mu_values = self.mu_values
		gamma_values = self.chromosome.population.gamma_values
		distal_intervals_n = self.left_intervals_n
		inversion_intervals_n = self.inversion_intervals_n
		proximal_intervals_n = self.proximal_intervals_n
		right_intervals = self.right_intervals_n
		intervals_n = self.intervals_n
		alpha = self.population.alpha
		if self.chromosome.statespace is None:
			self.generate_statespace()
		self.statespace = self.chromosome.statespace
		if self.chromosome.transition_matrices is None:
			self.generate_transition_matrices() 
		
		if self.chromosome.population.g_values is None:
			self.chromosome.population.generate_g_values(200)
		
		self.g_values = self.chromosome.population.g_values
		
		self.generate_matrices()
		if self.chromosome.population.pi_vector is None:
			self.chromosome.population.generate_pi_vector()
		pi_vector = self.chromosome.population.pi_vector
		self.pi_vector = pi_vector # stationary phase distribution
		
		start = [False for i in range(self.intervals_n)]
		start.append(0)
		v0 = numpy.zeros(len(self.statespace))
		start_index = self.find_indices(start, self.statespace)[0]
		v0[start_index] = 1.0
		dot = numpy.dot
		v = numpy.zeros(len(self.statespace), numpy.float64)
		if self.chromosome.Q is None:
			self.generate_Q()
		Q = self.chromosome.Q # Q matrix
		truncate = self.new_truncate_D # truncate (list) gives the upper limit on the number of chiasma events to be considered in each interval. See method generate_matrices
		args = [range(truncate[i]) for i in range(self.intervals_n)]
		chiasmata_iter = itertools.product(*args)
		alpha = self.chromosome.population.alpha
		left_intervals_n = self.left_intervals_n
		inversion_intervals_n = self.inversion_intervals_n
		intervals_n = self.intervals_n
		
		Ds = self.Ds # Ds[i][x] is the D matrix for interval i, x chiasma events
		Ps = self.Ps # Ps[i][x] is the P matrix for interval i to the xth power.
		
		for chiasmata_combination in chiasmata_iter: # equivalent to a nested for-loop over all number of chiasma events in each interval.
			p = 0.0
			mid_vector_p = pi_vector
			mid_vector_s = v0
			for i in range(left_intervals_n):
				x = chiasmata_combination[i]
				mid_vector_p = dot(mid_vector_p, Ds[i][x])
				mid_vector_s = dot(mid_vector_s, Ps[i][x])
			if not alpha:
				mid_vector_p = dot(mid_vector_p, Q)
			for i in range(left_intervals_n, left_intervals_n + inversion_intervals_n):
				x = chiasmata_combination[i]
				mid_vector_p = dot(mid_vector_p, Ds[i][x])
				mid_vector_s = dot(mid_vector_s, Ps[i][x])
			if not alpha:
				mid_vector_p = dot(mid_vector_p, Q)
			for i in range(left_intervals_n + inversion_intervals_n, intervals_n):
				x = chiasmata_combination[i]
				mid_vector_p = dot(mid_vector_p, Ds[i][x])
				mid_vector_s = dot(mid_vector_s, Ps[i][x])
			v += numpy.sum(mid_vector_p)*mid_vector_s
			
		
		self.v = v
		
		is_balanced = self.is_balanced
		find_indices = self.find_indices
		
		tf = []
		for i in range(intervals_n):
			tf.append([True, False])
		patterns = {}
		for pattern in itertools.product(*tf):
			if is_balanced(pattern): # balanced
				w = numpy.zeros(len(self.statespace), numpy.float64)
				state = list(pattern)
				state = state + ['x']
				pattern_indices = find_indices(state, self.statespace)
				w[pattern_indices] = 1.0
				state_a1 = copy.copy(state)
				state_a1[-1] = 1
				pattern_a1_indices = find_indices(state_a1, self.statespace)
				w[pattern_a1_indices] = 2.0 # balanced chromatids in an anaphase I tetrad configuration are counted twice.
				p = numpy.dot(v, w)
				patterns[pattern] = p

		w = numpy.zeros(len(self.statespace), numpy.float64)
		unbalanced_indices = find_indices('unbalanced', self.statespace)
		w[unbalanced_indices] = 1.0
		unbalanced_a1_indices = find_indices('unbalanced_a1', self.statespace)
		w[unbalanced_a1_indices] = 0.0 # unbalanced chromatids in an anaphase I tetrad are retained in the polar bodies.
		unbalanced_p = numpy.dot(v, w)

		patterns['unbalanced'] = unbalanced_p
		
		self.recombination_patterns = patterns
		
	def calculate_recombination_pattern_probabilities_2(self):
		'''
		Implements theorems 1/2/5, depending on circumstances.
		'''
		self.generate_GHs()
		self.generate_RNs()
		if self.chromosome.population.pi_vector is None:
			self.chromosome.population.generate_pi_vector()
		self.pi_vector = self.chromosome.population.pi_vector # stationary phase distribution
		if not self.homokaryotype and self.chromosome.Q is None:
			self.generate_Q()
		self.Q = self.chromosome.Q # Q matrix
		pi_vector = self.pi_vector
		recombination_patterns = {}
		left_intervals_n = self.left_intervals_n
		inversion_intervals_n = self.inversion_intervals_n
		intervals_n = self.intervals_n
		dot = numpy.dot
		# Matrices R and N are the same as matrix M in thesis, for the two conditions.
		Rs = self.Rs # Rs[i] give matrix R for interval i, where matrix R is equal to 1/2 times matrix G
		Ns = self.Ns # Ns[i] give matrix N for interval i, where matrix N is equal to 1/2 times matrix G plus matrix H
		if not self.homokaryotype:
			Q = self.Q
		alpha = self.population.alpha
		patterns = {'unbalanced': 0.0}
		tf = [[False, True] for i in range(intervals_n)]
		
		for pattern in itertools.product(*tf): # equivalent to a for-loop over all possible patterns.
			mid_vector = pi_vector
			for i in range(left_intervals_n):
				if pattern[i]:
					mid_vector = dot(mid_vector, Rs[i])
				else:
					mid_vector = dot(mid_vector, Ns[i])
	
			if not alpha and not self.homokaryotype:
				mid_vector = dot(mid_vector, Q)
			for i in range(left_intervals_n, left_intervals_n +inversion_intervals_n):
				if pattern[i]:
					mid_vector = dot(mid_vector, Rs[i])
				else:
					mid_vector = dot(mid_vector, Ns[i])
			if not alpha and not self.homokaryotype:
				mid_vector = dot(mid_vector, Q)
			for i in range(left_intervals_n+inversion_intervals_n, intervals_n):
				if pattern[i]:
					mid_vector = dot(mid_vector, Rs[i])
				else:
					mid_vector = dot(mid_vector, Ns[i])
			prob = sum(mid_vector)
			if sum(pattern[left_intervals_n: left_intervals_n+inversion_intervals_n])%2 != 0: # unbalanced pattern
				patterns['unbalanced'] += prob
			else:
				patterns[pattern] = prob
		self.recombination_patterns = patterns

		
	def generate_matrices(self):
		'''
		Generates and stores matrices D(x) and P^x for each interval.
		'''
		new_truncate = []
		lambda_values = self.lambda_values
		mu_values = self.mu_values
		g_values = self.chromosome.population.g_values
		pi_vector = self.chromosome.population.pi_vector
		matrix_power = self.chromosome.population.calculator.matrix_power
		error = self.chromosome.population.error2 # error is defined so that the recombination patterns sum to approximately 1-error.
		Ds = [[] for i in range(self.intervals_n)]
		Ps = [[] for i in range(self.intervals_n)]
		for i in range(self.intervals_n):
			s = 0.0
			x = 0
			while True:
				while g_values.shape[0] < x*(self.chromosome.population.m+1): # extend g values matrix if needed.
					self.population.extend_g_values(10)
					g_values = self.population.g_values
				D = self.generate_D(x, lambda_values[i], mu_values[i], g_values)
				P = matrix_power(self.chromosome.transition_matrices[i], x)
				Ds[i].append(D)
				Ps[i].append(P)
				s += sum(pi_vector.dot(D))
				if s > 1-error:
					new_truncate.append(x+1)
					break
				x += 1
		self.new_truncate_D = new_truncate
		self.Ds = Ds
		self.Ps = Ps
		
	
	def generate_D(self, x, lambda_value, mu_value, g_values):
		'''
		Generates and returns a single D(x) matrix as defined in the thesis.
		
		Arguments:
		x: (int) the number of chiasma events
		lambda_value/mu_value/g_value: see class Chromosome
		'''
		m = self.chromosome.population.m
		matrix = numpy.zeros((m+1,m+1), numpy.float64)
		if lambda_value+mu_value != 0:
			p1 = mu_value/(lambda_value+mu_value)
			p2 = lambda_value/(lambda_value+mu_value)
		else:
			p1 = 0.0
			p2 = 0.0
		comb = scipy.misc.comb
		gamma_values = self.chromosome.population.gamma_values
		poisson = self.chromosome.population.calculator.poisson
		if x == 0:
			for i in range(m+1):
				for j in range(m+1):
					if i>=j:
						matrix[i][j] = poisson(lambda_value, i-j)*exp(-mu_value)
		else:
			for i in range(m+1):
				for j in range(m+1):
					s = 0.0
					for l in range(x):
						for n in range(x-l-1, (x-l-1)*(m+1)+1):
							for q in range(j, m+1):
								h = i+1+l+n+q-j
								s += g_values[n][x-l-1]*gamma_values[q]*( (exp(-(lambda_value+mu_value))*(lambda_value+mu_value)**h)/factorial(h) ) * comb(h,l)*(p1**l)*(p2**(h-l))
					if i>=j:
						s += ( (exp(-mu_value)*mu_value**x)/factorial(x) ) * ( (exp(-lambda_value)*lambda_value ** (i-j) )/factorial(i-j) )
					matrix[i][j] = s
		
		return matrix
		
	def is_balanced(self, state):
		'''
		Returns True if state is balanced.
		'''
		return sum(state[self.left_intervals_n:self.left_intervals_n + self.inversion_intervals_n])%2 == 0
		
		
	def generate_statespace(self):
		'''
		Generates the statespace as defined in the thesis.
		'''
		distal_intervals_n = self.left_intervals_n
		inversion_intervals_n = self.inversion_intervals_n
		proximal_intervals_n = self.proximal_intervals_n
		opposite_intervals_n = self.right_intervals_n
		intervals_n = self.intervals_n
		is_balanced = self.is_balanced
		
		rec_mid = []
		statespace = []
		if inversion_intervals_n == 0:
			for i in range(distal_intervals_n):
				rec_mid.append([False,True])
			for r in itertools.product(*rec_mid):
				state = list(r)
				state_copy = copy.copy(state)
				state_copy.append(0)
				statespace.append(state_copy)
		else:	
			for i in range(distal_intervals_n + inversion_intervals_n + proximal_intervals_n + opposite_intervals_n):
				rec_mid.append([False, True])
			for r in itertools.product(*rec_mid): # loops over all possible recombination patterns, r
				state = list(r)
				for j in range(4): # j indicate the tetrad configuration as follows. 0: no bridge; 1: single a1 bridge; 2: single a2 bridge; 3: double bridge
					if j == 0: # no bridge
						if is_balanced(state): # only balanced chromatids can be in configuration no bridge
							state_copy = copy.copy(state)
							state_copy.append(j)
							statespace.append(state_copy)
					elif j == 1 or j == 2: # single a1 or a2 bridge
						state_copy = copy.copy(state)
						if not is_balanced(state):
							for k in range(distal_intervals_n + inversion_intervals_n, intervals_n): # ignore unbalanced chromatids with recombination in intervals to the right of the inversion
								state_copy[k] = False
						state_copy.append(j)
						if state_copy not in statespace:
							statespace.append(state_copy)

					else: # double bridge
						state_copy = copy.copy(state)
						if not is_balanced(state): # only unbalanced chromatids can be in configuration double bridge
							for k in range(distal_intervals_n + inversion_intervals_n, intervals_n):
								state_copy[k] = False
							state_copy.append(j)
							if state_copy not in statespace:
								statespace.append(state_copy)

	
		self.chromosome.statespace = statespace
	
	def generate_transition_matrices(self):
		'''
		Generates matrix P for each interval, as defined and discussed in the thesis.
		'''
		distal_intervals_n = self.left_intervals_n
		inversion_intervals_n = self.inversion_intervals_n
		proximal_intervals_n = self.proximal_intervals_n
		opposite_intervals_n = self.right_intervals_n
		intervals_n = self.intervals_n
		is_balanced = self.is_balanced
		statespace = self.chromosome.statespace
		matrices = []
		for i in range(distal_intervals_n): # Distal matrices
			matrix = numpy.zeros((len(statespace), len(statespace)))
			for state_from in statespace:
				if sum(state_from[i+1:intervals_n]) == 0 and state_from[-1] == 0: # ignore states that show recombination in more proximal intervals
					from_index = statespace.index(state_from)
				
					state_to1 = copy.copy(state_from)
					state_to2 = copy.copy(state_from)	
					
					state_to1[i] = True
					state_to2[i] = False
					
					to_index1 = statespace.index(state_to1)
					to_index2 = statespace.index(state_to2)
					
					matrix[from_index][to_index1] = 0.5
					matrix[from_index][to_index2] = 0.5
		
			matrices.append(matrix)
			
		for i in range(distal_intervals_n ,distal_intervals_n + inversion_intervals_n): # Inversion matrices
			matrix = numpy.zeros((len(statespace),len(statespace)))
			for state_from in statespace:
				if sum(state_from[i+1:intervals_n]) == 0: # ignore states that show recombination in more proximal intervals
					if state_from[-1] == 0: # no bridge
						state_to1 = copy.copy(state_from)
						state_to2 = copy.copy(state_from)
						state_to1[i] = True #1
						state_to2[i] = False #1
						state_to1[-1] = 1
						state_to2[-1] = 1
						from_index = statespace.index(state_from)
						to_index1 = statespace.index(state_to1)
						to_index2 = statespace.index(state_to2)
						matrix[from_index][to_index1] = 0.5
						matrix[from_index][to_index2] = 0.5
					elif state_from[-1] == 1: # single a1 bridge
						state_to1 = copy.copy(state_from)
						state_to2 = copy.copy(state_from)
						state_to3 = copy.copy(state_from)
						state_to4 = copy.copy(state_from)
			
						if is_balanced(state_from): #balanced
							if state_from[i] == False:
								state_to1[i] = False #2
								state_to1[-1] = 0
				
								state_to2[i] = True #3
								state_to2[-1] = 1
				
								state_to3[i] = False #3
								state_to3[-1] = 1
				
								state_to4[i] = True #4
								state_to4[-1] = 3
							else:
								state_to1[i] = True #2
								state_to1[-1] = 0
				
								state_to2[i] = True #3
								state_to2[-1] = 1
				
								state_to3[i] = False #3
								state_to3[-1] = 1
				
								state_to4[i] = False #4
								state_to4[-1] = 3
					
						else: #unbalanced
							if state_from[i] == False:
								state_to1[i] = True #2
								state_to1[-1] = 0
					
								state_to2[i] = True #3
								state_to2[-1] = 1
					
								state_to3[i] = False #3
								state_to3[-1] = 1
					
								state_to4[i] = False #4
								state_to4[-1] = 3
					
							else:
								state_to1[i] = False #2
								state_to1[-1] = 0
					
								state_to2[i] = False #3
								state_to2[-1] = 1
					
								state_to3[i] = True #3
								state_to3[-1] = 1
					
								state_to4[i] = True #4
								state_to4[-1] = 3
			
						from_index = statespace.index(state_from)
						to_index1 = statespace.index(state_to1)
						to_index2 = statespace.index(state_to2)
						to_index3 = statespace.index(state_to3)
						to_index4 = statespace.index(state_to4)
			
						matrix[from_index][to_index1] = 0.25
						matrix[from_index][to_index2] = 0.25
						matrix[from_index][to_index3] = 0.25
						matrix[from_index][to_index4] = 0.25
			
			
					elif state_from[-1] == 3: # double bridge
						state_to1 = copy.copy(state_from)
						state_to2 = copy.copy(state_from)
						state_to1[i] = True #5
						state_to1[-1] = 1
						state_to2[i] = False #5
						state_to2[-1] = 1
			
						from_index = statespace.index(state_from)
			
						to_index1 = statespace.index(state_to1)
						to_index2 = statespace.index(state_to2)
			
						matrix[from_index][to_index1] = 0.5
						matrix[from_index][to_index2] = 0.5
					
			matrices.append(matrix)
			
		for i in range(distal_intervals_n + inversion_intervals_n, distal_intervals_n + inversion_intervals_n + proximal_intervals_n): # proximal matrices
			matrix = numpy.zeros((len(statespace),len(statespace)))
			for state_from in statespace:
				from_index = statespace.index(state_from)
				if sum(state_from[i+1:intervals_n]) == 0: # ignore states that show recombination in more proximal intervals
					if state_from[-1] == 0: # no bridge
						state_to1 = copy.copy(state_from)
						state_to2 = copy.copy(state_from)
						state_to1[i] = True #1
						state_to2[i] = False #1
				
						to_index1 = statespace.index(state_to1)
						to_index2 = statespace.index(state_to2)
				
						matrix[from_index][to_index1] = 0.5
						matrix[from_index][to_index2] = 0.5
			
					elif state_from[-1] == 1: #single a1 bridge
						if is_balanced(state_from): # balanced
							state_to1 = copy.copy(state_from)
							state_to2 = copy.copy(state_from)
							state_to3 = copy.copy(state_from)
							state_to4 = copy.copy(state_from)
							if state_from[i] == False:
						
								state_to1[i] = True #5
								state_to1[-1] = 2
						
								state_to2[i] = True #4
								state_to2[-1] = 1
						
								state_to3[i] = False #4
								state_to3[-1] = 1
						
								state_to4[i] = False #5
								state_to4[-1] = 2
						
							else:
						
								state_to1[i] = False #5
								state_to1[-1] = 2
						
								state_to2[i] = False #4
								state_to2[-1] = 1
						
								state_to3[i] = True #4
								state_to3[-1] = 1
						
								state_to4[i] = True #5
								state_to4[-1] = 2
					
							to_index1 = statespace.index(state_to1)
							to_index2 = statespace.index(state_to2)
							to_index3 = statespace.index(state_to3)
							to_index4 = statespace.index(state_to4)
				
							matrix[from_index][to_index1] = 0.25
							matrix[from_index][to_index2] = 0.25
							matrix[from_index][to_index3] = 0.25
							matrix[from_index][to_index4] = 0.25
						
						else: #unbalanced
						
							state_to1 = copy.copy(state_from)
							state_to2 = copy.copy(state_from)
					
							# ignore recombination in the proximal region, as states will be unbalanced anyway
							state_to1[-1] = 1 #2
							state_to2[-1] = 2 #3

					
							to_index1 = statespace.index(state_to1)
							to_index2 = statespace.index(state_to2)
					
							matrix[from_index][to_index1] = 0.5 
							matrix[from_index][to_index2] = 0.5
						
				
					elif state_from[-1] == 2: # single a2 bridge
						if is_balanced(state_from):
							state_to1 = copy.copy(state_from)
							state_to2 = copy.copy(state_from)
							if state_from[i] == False:
								state_to1[i] = True #8
								state_to1[-1] = 1
						
								state_to2[i] = False #8
								state_to2[-1] = 1
					
							else:
								state_to1[i] = False #8
								state_to1[-1] = 1
						
								state_to2[i] = True #8
								state_to2[-1] = 1
					
							to_index1 = statespace.index(state_to1)
							to_index2 = statespace.index(state_to2)
				
							matrix[from_index][to_index1] = 0.5
							matrix[from_index][to_index2] = 0.5
					
					
					
						else: #unbalanced
							state_to = copy.copy(state_from)
							state_to[-1] = 1 #6
					
							to_index = statespace.index(state_to)
							matrix[from_index][to_index] = 1.0 # unbalanced gamete no matter what happens in the proximal region.
					
							
					else: # double bridge
						state_to = copy.copy(state_from) #9
						to_index = statespace.index(state_to)
						matrix[from_index][to_index] = 1.0 # unbalanced gamete no matter what happens in the proximal region.
			matrices.append(matrix)
			
		for i in range(distal_intervals_n + inversion_intervals_n + proximal_intervals_n, intervals_n): # opposite arm matrices
			matrix = numpy.zeros((len(statespace),len(statespace)))
			for state_from in statespace:
				if sum(state_from[i+1:intervals]) == 0: # ignore states that show recombination in intervals to the right of current interval
				
					if is_balanced(state_from):
						state_to1 = copy.copy(state_from)
						state_to2 = copy.copy(state_from)
					
						state_to1[i] = False
						state_to2[i] = True
					
						from_index = statespace.index(state_from)
						to_index1 = statespace.index(state_to1)
						to_index2 = statespace.index(state_to2)
					
						matrix[from_index][to_index1] = 0.5
						matrix[from_index][to_index2] = 0.5
						
						
						
					else: #unbalanced
						state_to = copy.copy(state_from)
						from_index = statespace.index(state_from)
						to_index = statespace.index(state_to)
						matrix[from_index][to_index] = 1.0 # unbalanced gamete no matter what happens in the opposite arm.
						
		
			matrices.append(matrix)
		
		self.chromosome.transition_matrices = matrices
		
			

	
	def generate_Q(self):
		'''
		Generates and stores the Q matrix, as defined in the thesis.
		'''
		Q = []
		pi_vector = list(self.chromosome.population.pi_vector)
		for i in range(len(pi_vector)):
			Q.append(pi_vector)
		Q = numpy.array(Q)
		self.Q = Q
		self.chromosome.Q = Q
	
	def generate_RNs(self):
		'''
		Generates and stores matrices R and N for each interval. These are the M matrices as defined in the thesis, for the two conditions (recombination and non-recombination)
		'''
		Rs = []
		Ns = []
		# self.GHs[i][0] and self.GHs[i][1] are matrices G and H, respectively, for interval i.
		for i in range(len(self.GHs)):
			Rs.append(0.5*self.GHs[i][0])
			Ns.append(0.5*self.GHs[i][0]+self.GHs[i][1])
		self.Ns = Ns
		self.Rs = Rs
		
	def generate_GHs(self):
		'''
		Generates and store matrices G and H for each interval, as defined in the thesis.
		'''
		GHs = []
		error = self.population.error1 # error is defined so that the recombination patterns sum to approximate 1-error when the infinite series form of matrix G is used.
		poisson = self.population.calculator.poisson
		intervals_n = self.chromosome.intervals_n
		closed_form = self.population.closed_form # True if the closed form of matrix G (theorem 2 in the thesis) is to be used.
		for k in range(intervals_n):
			lambda_value = self.lambda_values[k]
			mu_value = self.mu_values[k]
			gamma_values = self.population.gamma_values
			m = self.population.m
			G = numpy.zeros((m+1,m+1))
			H = numpy.zeros((m+1,m+1))
			f = self.population.calculator.f # function f in theorem 2 in the thesis.
			if gamma_values[m] == 1.0 and closed_form: # use theorem 2
				for i in range(m+1):
					for j in range(m+1):
						if i>=j:
							pois_term = poisson(lambda_value, i-j)
							psi = exp(-lambda_value)*f(i-j, m+1, lambda_value)-pois_term
						else:
							pois_term = 0.0
							psi = exp(-lambda_value)*f(m+1+i-j, m+1, lambda_value)
						G[i][j] = exp(-mu_value)*psi+(1-exp(-mu_value))*(psi+pois_term)
						H[i][j] = pois_term*exp(-mu_value)
			else:
				if len(self.population.b_values) == 1:
					self.population.extend_b_values(100)
				b_values = self.population.b_values
				for i in range(m+1):
					for j in range(m+1):
						psi = 0.0
						n = 0
						check = 0.0
						while check < 1-error:
							while len(b_values) <= n:
								self.population.extend_b_values()
								b_values = self.population.b_values

							for q in range(j, m+1):
								try:
									pois_factor = poisson(lambda_value, i+1+n+q-j)
								except OverflowError:
									print('Error! error1 is too small! Please choose a higher value for parameter error1 under heading # population')
									sys.exit(1)
								add = b_values[n]*gamma_values[q]*pois_factor
								psi += add
							check += poisson(lambda_value, n)
							n += 1
															
						if i>=j:
							pois_term = poisson(lambda_value, i-j)
						else:
							pois_term = 0.0
						G[i][j] = exp(-mu_value)*psi+(1-exp(-mu_value))*(psi+pois_term)
						H[i][j] = pois_term*exp(-mu_value)
			GHs.append([G,H])
	
		self.GHs = GHs
	
	
class Chromosome:
	'''
	A class for storing and manipulating information relating to a single chromosome
	
	Key attributes:
	self.key: (str) A string identifying the chromosome (can be any string without #, %, and spaces)
	self.loci: List of instances of class Locus
	self.loci_keys: A list of locus keys (single charachters)
	self.allele_ns_dictionary: A dictionary of allele numbers for each locus. e.g. {'A': 2, 'B':3} indicate that Locus A has two alleles and locus B has three alleles.
		If a locus is not listed, then the allele number is set to 2 by default.
	self.lambda_values/mu_values/d_values: (list) The lambda/mu/d values as defined in the thesis. The values are sorted so that index i of the list
		correspond to interval i.
	self.sex: (bool) True if the chromosome is the sex chromosome (i.e. if it contains the sex determination locus). See also subclass Sex_chromosome
	self.haplotypes/diplotypes/karyotypes: A list of Chromosome_haplotype/Chromosome_diplotype/Karyotype instances
	self.inversion: (bool) True if the chromosome has an inversion polymorphism
	self.Q: The Q matrix as defined in the thesis
	self.transition_matrices: The P matrices as defined in the thesis (theorem 7)
	self.intervals_n: (int) The number of intervals on the chromosome
	self.left_intervals_n: (int) The number of intervals to the left of the left inversion breakpoint, if self.inversion is True (equal to self.intervals_n if self.inversion is False)
	self.inversion_intervals_n: (int) The number of intervals in the inverted region
	self.proximal_intervals_n: (int) The number of intervals in the proximal region
	self.right_intervals_n: (int) The number of intervals in the region to the right of the proximal region if self.inversion is True and self.proximal_intervals_n>0
		or to the right of the inverted region if self.inversion is True and self.proximal_intervals_n == 0. Zero if self.inversion is False.
	self.original_homokaryotype/reversed_homokaryotype: If the chromosome has an inversion, then the order of the loci inside the inverted region will be reversed
		in the derived homokaryotype. self.original_homokaryotype points to the Karyotype instance with the original order, and self.reversed_homokaryotype to the
		Karyotype instance with the reversed order.
	self.original_heterokaryotype/reversed_heterokaryotype: 'original' and 'reversed' here refers to the direction of the interference signal through the
		inversion loop (see figure 2.3 in the thesis)
	
	
	
	
	
	'''
	def __init__(self, population, key, loci_keys, allele_ns_dictionary, lambda_values, mu_values, d_values, sex, index):
		self.population = population
		self.key = key
		self.loci_keys = loci_keys
		self.loci_keys_reversed = None
		self.loci_n = len(self.loci_keys)
		self.allele_ns_dictionary = allele_ns_dictionary
		if len(lambda_values) == 1 and self.loci_n != 2:
			self.lambda_values = [lambda_values[0] for i in range(self.loci_n-1)] # If only one lambda value is given, assume the user want to use this value for all intervals
		elif len(lambda_values) != self.loci_n - 1:
			print('Error! The number of lambda values for chromosome ' + self.key + ' does not correspond to the number of intervals!')
			sys.exit(1)
		
		else:
			self.lambda_values = lambda_values
		
		if len(mu_values) == 1 and self.loci_n != 2:
			self.mu_values = [mu_values[0] for i in range(self.loci_n-1)]
		elif len(mu_values) != self.loci_n - 1:
			print('Error! The number of mu values for chromosome ' + self.key + ' does not correspond to the number of intervals!')
			sys.exit(1)
		
		else:
			self.mu_values = mu_values
		
		
		self.d_values = d_values
		if sum(self.d_values) == 0: # no recombination in heterokaryotypes
			self.heteromorphic = True
		else:
			self.heteromorphic = False
		self.sex = sex
		self.loci = []
		self.haplotypes = []
		self.diplotypes = []
		self.diplotype_keys = []
		self.paracentric = False
		self.karyotypes = []
		self.heterokaryotypes = []
		self.homokaryotypes = []
		self.statespace = None
		self.transition_matrices = None
		self.Q = None

		if '[' in self.loci_keys and ']' in self.loci_keys:
			self.inversion = True
			self.left_breakpoint_index = self.loci_keys.index('[')
			self.right_breakpoint_index = self.loci_keys.index(']')
			self.loci_keys_reversed = self.loci_keys[0:self.left_breakpoint_index+1]+self.loci_keys[self.left_breakpoint_index+1:self.right_breakpoint_index][::-1]+self.loci_keys[self.right_breakpoint_index:]
			if '@' in self.loci_keys:
				self.centromere_index = self.loci_keys.index('@')
				self.population.linear_meiosis = True
				self.population.male_recombinant = False
				self.paracentric = True
				if self.centromere_index<self.left_breakpoint_index: # make sure that the proximal region is to the right of the inverted region
					self.loci_keys = self.loci_keys[::-1]
					self.lambda_values = self.lambda_values[::-1]
					self.mu_values = self.mu_values[::-1]
					self.left_breakpoint_index = self.loci_keys.index('[')
					self.right_breakpoint_index = self.loci_keys.index(']')

			
			else:
				self.paracentric = False
		else:
			self.inversion = False
		
		if not self.inversion:
			self.left_intervals_n = self.loci_n-1
			self.intervals_n = self.left_intervals_n
			self.inversion_intervals_n = 0
			self.proximal_intervals_n = 0
			self.right_intervals_n = 0
		else:
			if not self.paracentric:
				self.left_intervals_n = self.left_breakpoint_index
				self.inversion_intervals_n = self.right_breakpoint_index - self.left_breakpoint_index
				self.proximal_intervals_n = 0
				self.right_intervals_n = self.loci_n - 1 - self.right_breakpoint_index
			else:
				self.left_intervals_n = self.left_breakpoint_index
				self.inversion_intervals_n = self.right_breakpoint_index - self.left_breakpoint_index
				self.proximal_intervals_n = self.centromere_index - self.right_breakpoint_index
				self.right_intervals_n = self.loci_n - 1 - self.centromere_index
		
		self.intervals_n = self.left_intervals_n + self.inversion_intervals_n + self.proximal_intervals_n + self.right_intervals_n
		
		
		self.original_homokaryotype = Karyotype(self, original = True, homokaryotype = True)
		self.homokaryotypes.append(self.original_homokaryotype)
		self.karyotypes.append(self.original_homokaryotype)
		if not self.sex:
			if self.inversion:
				self.original_heterokaryotype = Karyotype(self, original = True, homokaryotype = False)
				self.reversed_homokaryotype = Karyotype(self, original = False, homokaryotype = True)
				self.heterokaryotypes.append(self.original_heterokaryotype)
				self.homokaryotypes.append(self.reversed_homokaryotype)
				self.karyotypes.append(self.original_heterokaryotype)
				self.karyotypes.append(self.reversed_homokaryotype)
				if not self.heteromorphic and self.population.alpha:
					self.reversed_heterokaryotype = Karyotype(self, original = False, homokaryotype = False)
					self.heterokaryotypes.append(self.reversed_heterokaryotype)
					self.karyotypes.append(self.reversed_heterokaryotype)

			
			
	def calculate_recombination_pattern_probabilities(self):
		print('Calculating recombination pattern probabilities for chromosome ', self.key)
		for k in self.karyotypes:
			k.calculate_recombination_pattern_probabilities()
		
	
	def initialize_loci(self):
		'''
		Set the default allele numbers for the special loci '[' and ']' (left and right breakpoints) and '@' (centromere),
		and initialize and store pointers to all loci.
		'''
		for i in range(len(self.loci_keys)):
			locus_key = self.loci_keys[i]
			if locus_key == '[' or locus_key == ']':
				breakpoint = True
				if locus_key in self.allele_ns_dictionary:
					if self.allele_ns_dictionary[locus_key] != 2:
						print('Error! The number of alleles for a breakpoint cannot be different from 2!')
						sys.exit(1)
			else:
				breakpoint = False
			if locus_key in self.allele_ns_dictionary:
				allele_n = self.allele_ns_dictionary[locus_key]
			else: # the default number of alleles is 2 for all loci except the centromere (@)
				if locus_key == '@':
					allele_n = 1
				else:
					allele_n = 2
			locus = Locus(chromosome = self, key = locus_key, allele_n = allele_n, macrolinked = False, microlinked = False, population = self.population, chromosome_index = i, breakpoint = breakpoint)
			locus.initialize_alleles()
			self.loci.append(locus)
	
	def initialize_haplotypes(self):
		'''
		Initialize all possible chromosome haplotypes, except the ones that the user wants removed (as indicated in the input file)
		'''
		allele_keys = []
		for locus in self.loci:
			new_allele_keys = []
			for allele in locus.alleles:
				new_allele_keys.append(allele.allele_key)
			allele_keys.append(new_allele_keys)
		iterator = itertools.product(*allele_keys)
		for haplotype_allele_keys in iterator: # loop over all possible haplotypes
			passed = True
			if self.inversion:
				if haplotype_allele_keys[self.left_breakpoint_index] != haplotype_allele_keys[self.right_breakpoint_index]: # the two inversion breakpoints must belong to the same arrangement (ancestral/derived), otherwise the haplotype is unbalanced.
					passed = False
			if passed:
				haplotype = Chromosome_haplotype(allele_keys=haplotype_allele_keys, chromosome=self)
				haplotype.find_alleles()
				haplotype.find_initial_frequencies()
				for haplotype_key in self.population.haplotypes_to_remove: # remove haplotype if indicated by the user.
					haplotype_keys_set = set(haplotype_key.split('&'))
					if haplotype_keys_set.issubset(haplotype.locus_allele_keys):
						passed = False
						print('Haplotype removed:', haplotype.locus_allele_keys)
						break
				if passed:
					self.haplotypes.append(haplotype)
					self.population.haplotypes.append(haplotype)
	
	def initialize_diplotypes(self):
		for haplotype_pair in itertools.product(*[self.haplotypes, self.haplotypes]): # cartesian product of all haplotypes
			haplotype_pair = sorted(haplotype_pair)
			male = None	
			if self.inversion:
				if haplotype_pair[0].allele_keys[self.left_breakpoint_index] == 0 and haplotype_pair[1].allele_keys[self.left_breakpoint_index]	 == 0: # ancestral homokaryotype
					karyotypes = [self.original_homokaryotype]
					original_homokaryotype = True
					reversed_homokaryotype = False
					heterokaryotype = False
				elif haplotype_pair[0].allele_keys[self.left_breakpoint_index] == 1 and haplotype_pair[1].allele_keys[self.left_breakpoint_index] == 1: # derived homokaryotype
					karyotypes = [self.reversed_homokaryotype]
					original_homokaryotype = False
					reversed_homokaryotype = True
					heterokaryotype = False
				elif haplotype_pair[0].allele_keys[self.left_breakpoint_index] == 0 and haplotype_pair[1].allele_keys[self.left_breakpoint_index] == 1: # heterokaryotype
					karyotypes = self.heterokaryotypes
					original_homokaryotype = False
					reversed_homokaryotype = False
					heterokaryotype = True
			else:
				karyotypes = [self.original_homokaryotype]
				original_homokaryotype = True
				reversed_homokaryotype = False
				heterokaryotype = False
			diplotype = Chromosome_diplotype(haplotype_pair, self, karyotypes, original_homokaryotype, reversed_homokaryotype, heterokaryotype, male = male) # create a Chromosome_diplotype instance with a pair of Chromosome_haplotype instances as an attribute.
			initial_frequencies = []
			for d in range(self.population.demes_n):
				initial_frequencies.append(haplotype_pair[0].initial_frequencies[d]*haplotype_pair[1].initial_frequencies[d]) # Hardy-Weinberg
			diplotype.set_initial_frequencies(initial_frequencies)
			diplotype_key = diplotype.make_key()
			if not diplotype_key in self.diplotype_keys:
				self.diplotypes.append(diplotype)
				self.diplotype_keys.append(diplotype_key)
			else:
				for i in range(self.population.demes_n):
					self.diplotypes[self.diplotype_keys.index(diplotype_key)].initial_frequencies[i] += diplotype.initial_frequencies[i]
	
			
class Chromosome_diplotype:
	'''
	A class for storing and manipulating information about a single chromosome diplotype (two chromosome haplotypes)
	
	Key attributes:
	self.haplotypes: A list of the two Chromosome_haplotype instances, sorted according to the Chromosome_haplotype function __lt__.
	self.karyotypes: A List of instances of class Karyotype.
	self.original_homokaryotype: (bool) True if the diplotype is homozygous for the ancestral arrangement.
	self.reversed_homokaryotype: (bool) True if the diplotype is homozygous for the derived arrangement.
	self.heterokaryotype: (bool) True if the diplotype is an inversion heterokaryotype
	self.chromosome: Pointer to the chromosome
	self.male: (bool) None if self.chromosome is not the sex chromosome. True if self.chromosome is the sex chromosome and the diplotype is male. False if
		self.chromosome is the sex chromosome and the diplotype is female.
	self.gametes: A dictionary of gamete frequencies. A gamete is here represented as a string of allele keys (integers), where the character at index i gives the allele key for locus with index i
		(as given by the ordering of the loci in the input file). E.g. if for the given chromosome, the input file reads 'Loci = ABC', then the gamete '001' indicate alleles A0, B0, and C1.
		Note that the ordering of the loci in the gametes is always the same as in the input file, even for derived homokaryotypes.
	
	
	
	'''
	def __init__(self, haplotypes, chromosome, karyotypes, original_homokaryotype = False, reversed_homokaryotype = False, heterokaryotype = False, male = None):
		self.haplotypes = haplotypes
		self.chromosome = chromosome
		self.karyotypes = karyotypes
		self.original_homokaryotype = original_homokaryotype
		self.reversed_homokaryotype = reversed_homokaryotype
		self.heterokaryotype = heterokaryotype
		self.male = male
		self.gametes = []
		self.male_indices = []
		self.female_indices = []
		self.nonrecombinant_gametes = None
	
	def add_index(self, index, male):
		if male:
			self.male_indices.append(index)
		else:
			self.female_indices.append(index)
		for haplotype in self.haplotypes:
			haplotype.add_index(index, male)
		
	def generate_nonrecombinant_gametes(self):
		'''
		When there is no recombination, simply copy the chromosome haplotypes to make gametes.
		'''
		self.nonrecombinant_gametes = {}
		if self.haplotypes[0].allele_keys == self.haplotypes[1].allele_keys:
			self.nonrecombinant_gametes[str(self.haplotypes[0].allele_keys).replace('(', '').replace(')','').replace(',','').replace(' ','')] = 1.0
		else:
			for haplotype in self.haplotypes:
				self.nonrecombinant_gametes[str(haplotype.allele_keys).replace('(', '').replace(')','').replace(',','').replace(' ','')] = 0.5
		
	
	def set_initial_frequencies(self, initial_frequencies):
		self.initial_frequencies = initial_frequencies
	
	def make_key(self):
		key = ''
		for i in range(len(self.chromosome.loci)):
			l = self.chromosome.loci[i]
			key += l.key
			key += str(self.haplotypes[0].allele_keys[i])
			key += str(self.haplotypes[1].allele_keys[i])
		self.key = key
		return key
	
	def print_initial_frequencies(self):
		print(self.key,':', self.initial_frequencies)
	
	
	def calculate_gamete_frequencies(self):
		'''
		Translates the recombination patterns (calculated in class Karyotype) into gametes.
		'''
		allele_keys_0 = str(self.haplotypes[0].allele_keys).replace(' ','').replace(',','').replace('(','').replace(')','').replace('\'','') # convert from tuple to string
		allele_keys_1 = str(self.haplotypes[1].allele_keys).replace(' ','').replace(',','').replace('(','').replace(')','').replace('\'','')
		reverse = self.chromosome.population.calculator.reverse
		for karyotype in self.karyotypes: 
			'''
			Loop over karyotypes. For homokaryotypes, self.karyotypes only have a single element. For heterokaryotypes, it has two, 
			corresponding to the two possible directions of the interference signal across inversion breakpoint boundaries (see figure 2.3 in the thesis). 
			If there is no interference across breakpoint boundaries (alpha = 0), these karyotypes give identical results.
			
			'''
			if not (karyotype.original or karyotype.heteromorphic): # reverse loci order in the inverted region if karyotype has the derived arrangement (reversed back again later)
				l = self.chromosome.left_breakpoint_index
				r = self.chromosome.right_breakpoint_index
				allele_keys_0 = reverse(allele_keys_0, l, r)
				allele_keys_1 = reverse(allele_keys_1, l, r)
			gametes = {}
			for pattern, frequency in karyotype.recombination_patterns.items():
				if pattern == 'unbalanced':
					if frequency != 0:
						gametes['unbalanced'] = frequency
				else:
					gamete_0 = allele_keys_0[0]
					gamete_1 = allele_keys_1[0]
					new_gamete_0 = ''
					new_gamete_1 = ''
					for i in range(len(pattern)):
						# build up gametes by appending from the appropriate homologue
						if pattern[i]: # pattern[i] is True if the pattern shows recombination in interval i, and False if it does not.
							new_gamete_0 = gamete_1 + allele_keys_0[i+1]
							new_gamete_1 = gamete_0 + allele_keys_1[i+1]
						else:
							new_gamete_0 = gamete_0 + allele_keys_0[i+1]
							new_gamete_1 = gamete_1 + allele_keys_1[i+1]
					
						gamete_0 = new_gamete_0
						gamete_1 = new_gamete_1
					if not (karyotype.original or karyotype.heteromorphic): # reverse back
						gamete_0 = reverse(gamete_0, l, r)
						gamete_1 = reverse(gamete_1, l, r)
					frequency = frequency/2.0 # The two gametes are equally likely
					if frequency > 0:
						if gamete_0 in gametes:
							gametes[gamete_0] += frequency
						else:
							gametes[gamete_0] = frequency
						if gamete_1 in gametes:
							gametes[gamete_1] += frequency
						else:
							gametes[gamete_1] = frequency
						
			
			self.gametes.append(gametes)
		
		
		if len(self.gametes) == 1:
			self.gametes = self.gametes[0]
		else:
			g = {}
			for key in self.gametes[0]:
				g[key] = (self.gametes[0][key]+self.gametes[1][key])/2.0
			self.gametes = g


				
			
class Chromosome_haplotype:
	'''
	A class for storing and manipulation information relating to a single chromosome haplotype.
	
	Key attributes:
	self.allele_keys: The tuple of allele keys (integers; sorted by locus index).
	self.chromosome: A pointer to the Chromosome instance.
	self.locus_allele_keys: A list of locus_allele_keys, where a single such key consist of a locus key (single character) and an allele key (single digit)
	self.key: (str) A string of concatenated locus_allele_keys for each locus in the order they appear on the chromsome. E.g. the key A0B0C1 indicate
		alleles A0, B0 and C1.

	
	'''
	def __init__(self, allele_keys, chromosome):
		self.allele_keys = allele_keys
		self.chromosome = chromosome
		self.male_indices = []
		self.female_indices = []
		self.genotype_indices = []
		self.make_keys()
	
	def make_keys(self):
		self.locus_allele_keys = []
		self.key = ''
		loci_keys = self.chromosome.loci_keys
		for i in range(len(loci_keys)):
			locus_allele_key = loci_keys[i]+str(self.allele_keys[i])
			self.key += locus_allele_key
			self.locus_allele_keys.append(locus_allele_key)
	
	def add_index(self, index, male):
		if male:
			self.male_indices.append(index)
		else:
			self.female_indices.append(index)
		for allele in self.alleles:
			allele.add_index(index, male)
	
	def find_alleles(self):
		alleles = []
		for i in range(len(self.allele_keys)):
			locus = self.chromosome.loci[i]
			allele_key = self.allele_keys[i]
			if allele_key == '-':
				allele = locus.alleles[-1]
			else:
				allele = locus.alleles[allele_key]
			alleles.append(allele)
		self.alleles = alleles
		
	def find_initial_frequencies(self):
		initial_frequencies = [1.0 for i in range(self.chromosome.population.demes_n)]
		for d in range(len(initial_frequencies)):
			for allele in self.alleles:
				initial_frequencies[d] *= allele.initial_frequencies[d] # Hardy-Weinberg
		self.initial_frequencies = initial_frequencies
	
	def __lt__(self, other):
		'''
		A function used for sorting Allele instances. Returns True if self is smaller than other. For two autosomal or two macrolinked haplotypes, the smaller instance
		is the one with the smaller tuple of allele keys when these are read as binary numbers. For one macrolinked and one microlinked haplotype, the latter is the smaller.
		'''
		if self.chromosome.sex and self.allele_keys[self.chromosome.sexlocus_index] != other.allele_keys[self.chromosome.sexlocus_index]:
			if self.allele_keys[self.chromosome.sexlocus_index] == 0:
				return True
			else:
				return False
		else:
			return self.allele_keys<other.allele_keys
		
			

class Sex_chromosome(Chromosome):
	'''
	A class for storing and manipulating information relating to a single sex chromosome (subclass of class Chromosome)
	
	Key attributes:
	self.male_heterogametic: (bool) True if males are heterogametic, False if females are heterogametic
	self.macrolinked_loci_keys: A list of keys (single-character strings) to loci linked to the macro sex chromosome homologue (X or Z)
	self.microlinked_loci_keys: A list of keys to loci linked to the micro sex chromosome homologue (Y or W)
	'''
	def __init__(self, population, key, loci_keys, allele_ns_dictionary, lambda_values, mu_values, d_values, sex, index, male_heterogametic, macrolinked_loci, microlinked_loci):
		Chromosome.__init__(self, population = population, key = key, loci_keys = loci_keys, allele_ns_dictionary = allele_ns_dictionary, lambda_values = lambda_values, mu_values = mu_values, d_values = d_values, sex = sex, index = index)
		self.male_heterogametic = male_heterogametic
		self.population.male_heterogametic = male_heterogametic
		self.population.set_sex_keys()
		self.macrolinked_loci_keys = macrolinked_loci
		self.macrolinked_loci_indices = []
		self.microlinked_loci_keys = microlinked_loci
		self.microlinked_loci_indices = []
		self.allele_ns_dictionary['$'] = 2
		error = False
		
		# If at least one loci is either micro- or macro-linked, then all loci must be:
		if len(self.macrolinked_loci_keys) > 0 or len(self.microlinked_loci_keys) > 0:
			self.heteromorphic = True
			if not set(self.macrolinked_loci_keys).issubset(set(loci_keys)):
				error = True
			elif not set(self.microlinked_loci_keys).issubset(set(loci_keys)):
				error = True
			elif len(self.macrolinked_loci_keys) + len(self.microlinked_loci_keys) != len(loci_keys)-1:
				error = True
		
		# If no loci are micro- or macro-linked, the program assumes the chromosomes are homomorphic and recombine normally:
		else:
			self.heteromorphic = False
		
		if error:
			print('Error! Input for micro/macrolinked loci do not match the given sex chromosome loci. Please revise.')
			sys.exit(1)
			
		
		
		for macrolinked_locus_key in self.macrolinked_loci_keys:
			self.macrolinked_loci_indices.append(self.loci_keys.index(macrolinked_locus_key))

		for microlinked_loci_key in self.microlinked_loci_keys:
			self.microlinked_loci_indices.append(self.loci_keys.index(microlinked_loci_key))
			
		
		self.sexlocus_index = self.loci_keys.index('$') # The sex determination locus must be given the key '$'
		
		if self.inversion:
			if self.left_breakpoint_index < self.sexlocus_index and self.right_breakpoint_index > self.sexlocus_index:
				self.sexlocus_in_inversion = True
			else:
				self.sexlocus_in_inversion = False
		
		if self.heteromorphic:
			self.heteromorphic_heterokaryotype = Karyotype(self, original = False, homokaryotype = False, heteromorphic = True)
			self.heterokaryotypes.append(self.heteromorphic_heterokaryotype)
			self.karyotypes.append(self.heteromorphic_heterokaryotype)
		else:
			if self.inversion:
				self.reversed_homokaryotype = Karyotype(self, original = False, homokaryotype = True)
				self.original_heterokaryotype = Karyotype(self, original = True, homokaryotype = False)
				self.homokaryotypes.append(self.reversed_homokaryotype)
				self.heterokaryotypes.append(self.original_heterokaryotype)
				self.karyotypes.append(self.reversed_homokaryotype)
				self.karyotypes.append(self.original_heterokaryotype)
				if self.population.alpha:
					self.reversed_heterokaryotype = Karyotype(self, original = False, homokaryotype = False)
					self.heterokaryotypes.append(self.reversed_heterokaryotype)
					self.karyotypes.append(self.reversed_heterokaryotype)
			
	
	def initialize_diplotypes(self):
		'''
		A version of the initialize_diplotypes method (see class Chromosome) that takes into account special circumstances related to sex chromosomes.
		'''
		for haplotype_pair in itertools.product(*[self.haplotypes, self.haplotypes]): # a loop over all haplotype pairs
			haplotype_pair = sorted(haplotype_pair)
			passed = True
			if haplotype_pair[0].allele_keys[self.sexlocus_index] != haplotype_pair[1].allele_keys[self.sexlocus_index]: # heterozygous at the sex determination locus
				if self.population.male_heterogametic:
					male = True
				else:
					male = False
			elif haplotype_pair[0].allele_keys[self.sexlocus_index] == 1 and haplotype_pair[1].allele_keys[self.sexlocus_index] == 1: # homozygous at the sex determination locus
				if self.population.male_heterogametic:
					male = False
				else:
					male = True
			else:
				passed = False # no diplotypes can have two microgametes (allele key 0)
				
			if passed:
				if self.inversion:
					if haplotype_pair[0].allele_keys[self.left_breakpoint_index] == 0 and haplotype_pair[1].allele_keys[self.left_breakpoint_index]	 == 0:
						karyotypes = [self.original_homokaryotype]
						original_homokaryotype = True
						reversed_homokaryotype = False
						heterokaryotype = False
					elif haplotype_pair[0].allele_keys[self.left_breakpoint_index] == 1 and haplotype_pair[1].allele_keys[self.left_breakpoint_index] == 1:
						karyotypes = [self.reversed_homokaryotype]
						original_homokaryotype = False
						reversed_homokaryotype = True
						heterokaryotype = False
					elif haplotype_pair[0].allele_keys[self.left_breakpoint_index] != haplotype_pair[1].allele_keys[self.left_breakpoint_index]:
						karyotypes = self.heterokaryotypes
						original_homokaryotype = False
						reversed_homokaryotype = False
						heterokaryotype = True
				
				elif self.heteromorphic:
					if haplotype_pair[0].allele_keys[self.sexlocus_index] == haplotype_pair[1].allele_keys[self.sexlocus_index]:
						karyotypes = [self.original_homokaryotype]
						original_homokaryotype = True
						reversed_homokaryotype = False
						heterokaryotype = False
					else:
						karyotypes = [self.heteromorphic_heterokaryotype]
						original_homokaryotype = False
						reversed_homokaryotype = False
						heterokaryotype = True
				
				else:
					karyotypes = [self.original_homokaryotype]
					original_homokaryotype = True
					reversed_homokaryotype = False
					heterokaryotype = False
				diplotype = Chromosome_diplotype(haplotype_pair, self, karyotypes, original_homokaryotype, reversed_homokaryotype, heterokaryotype, male)
				initial_frequencies = []
				for d in range(self.population.demes_n):
					initial_frequencies.append(haplotype_pair[0].initial_frequencies[d]*haplotype_pair[1].initial_frequencies[d]) # Hardy-Weinberg
				diplotype.set_initial_frequencies(initial_frequencies)
				diplotype_key = diplotype.make_key()
				if not diplotype_key in self.diplotype_keys:
					self.diplotypes.append(diplotype)
					self.diplotype_keys.append(diplotype_key)
				else:
					for i in range(self.population.demes_n):
						self.diplotypes[self.diplotype_keys.index(diplotype_key)].initial_frequencies[i] += diplotype.initial_frequencies[i]
				
	
	def initialize_loci(self):
		'''
		See corresponding method in class Chromosome.
		'''
		for i in range(len(self.loci_keys)):
			locus_key = self.loci_keys[i]
			microlinked = False
			macrolinked = False
			if locus_key in self.microlinked_loci_keys:
				microlinked = True
			elif locus_key in self.macrolinked_loci_keys:
				macrolinked = True
			
			if locus_key == '[' or locus_key == ']':
				breakpoint = True
				if locus_key in self.allele_ns_dictionary:
					if self.allele_ns_dictionary[locus_key] != 2:
						print('Error! The number of alleles for a breakpoint cannot be different from 2!')
						sys.exit(1)
			else:
				breakpoint = False
			if locus_key in self.allele_ns_dictionary:
				allele_n = self.allele_ns_dictionary[locus_key]
			else: # the default number of alleles is 2 for all loci except the centromere (@) 
				if locus_key == '@':
					allele_n = 1
				else:
					allele_n = 2
				print('NB! No input found for allele number for locus ' + locus_key + ' of chromosome ' + self.key +'! Using default value ' + str(allele_n)) 
			locus = Locus(chromosome = self, key = locus_key, allele_n = allele_n, macrolinked = macrolinked, microlinked = microlinked, population = self.population, chromosome_index = i, breakpoint = breakpoint)
			locus.initialize_alleles()
			self.loci.append(locus)
	
	def initialize_haplotypes(self):
		'''
		See corresponding method in class Chromosome.
		'''
		allele_keys = []
		for locus in self.loci:
			new_allele_keys = []
			for allele in locus.alleles:
				new_allele_keys.append(allele.allele_key)
			allele_keys.append(new_allele_keys)
		iterator = itertools.product(*allele_keys)
		for haplotype_allele_keys in iterator:
			passed = True
			if self.inversion:
				if not haplotype_allele_keys[self.left_breakpoint_index] == haplotype_allele_keys[self.right_breakpoint_index]: # the two inversion breakpoints must have the same orientation
					passed = False
				if passed:
					if self.heteromorphic and haplotype_allele_keys[self.sexlocus_index] != haplotype_allele_keys[self.left_breakpoint_index]:
						passed = False
			if passed:
				for microlinked_locus_index in self.microlinked_loci_indices:
					if haplotype_allele_keys[self.sexlocus_index] == 0 and haplotype_allele_keys[microlinked_locus_index] == '-': # microlinked loci cannot have dummy alleles (key '-') on the micro sex homologue
						passed = False
						break
					elif haplotype_allele_keys[self.sexlocus_index] == 1 and haplotype_allele_keys[microlinked_locus_index] != '-': # microlinked loci cannot have allele that are *not*  dummy alleles (key '-') on the macro sex homologue
						passed = False
						break
			if passed:
				for macrolinked_locus_index in self.macrolinked_loci_indices:
					if haplotype_allele_keys[self.sexlocus_index] == 0 and haplotype_allele_keys[macrolinked_locus_index] != '-': # macrolinked loci cannot have alleles that are *not* dummy alleles (key '-') on the micro sex homologue
						passed = False
						break
					elif haplotype_allele_keys[self.sexlocus_index] == 1 and haplotype_allele_keys[macrolinked_locus_index] == '-': # macrolinked loci cannot have dummy alleles (key '-') on the macro sex homologue
						passed = False
						break
			if passed:
				haplotype = Chromosome_haplotype(allele_keys=haplotype_allele_keys, chromosome=self)
				haplotype.find_alleles()
				haplotype.find_initial_frequencies()
				self.haplotypes.append(haplotype)
				self.population.haplotypes.append(haplotype)

class Locus:
	'''
	A class for storing and manipulating information about a single locus.
	
	Key attributes:
	self.chromsome: The chromosome instance
	self.key: (str) A unique single character identifying the locus
	self.alleles_n: (int) The number of alleles of the locus
	self.alleles: The list of Allele instances.
	self.allele_keys: The list of allele keys (integers).
	self.macrolinked: (bool) True if the locus is linked to the macro sex chromosome (X/Z).
	self.microlinked: (bool) True if the locus is linked to the micro sex chromosome (Y/W).
	self.sex: (bool) True if the locus is the sex determination locus.
	self.breakpoint: (bool) True if the locus is an inversion breakpoint
	self.chromosome_index: (int) The index of the locus among the loci on the same chromosome.
	
	'''
	
	
	def __init__(self, chromosome, key, allele_n, macrolinked, microlinked, population, chromosome_index, breakpoint):
		self.chromosome = chromosome
		self.key = key
		self.allele_n = allele_n
		self.alleles = []
		self.macrolinked = macrolinked
		self.microlinked = microlinked
		self.mitochondria = False # This function is not currently in use. In future versions of the program, I will allow for mitochondrial inheritance
		if self.key == '$':
			self.sex = True
			self.allele_n = 2
		else:
			self.sex = False
		self.population = population
		self.chromosome_index = chromosome_index 
		self.breakpoint = breakpoint
		self.allele_keys = []
	
	def initialize_alleles(self):
		'''
		Create and store the appropriate number of Allele instances
		'''
		for i in range(self.allele_n):
			allele = Allele(self, i, self.population) # new Allele instance
			self.alleles.append(allele)
			self.population.alleles.append(allele)
			self.allele_keys.append(i)
			self.population.alleles_dictionary[allele.key] = allele
		if self.microlinked or self.macrolinked:
			allele = Allele(self, '-', self.population) # If the allele is sex-linked, make a dummy allele that serves as a placeholder for the non-existing homologue.
			self.alleles.append(allele)
			self.population.alleles.append(allele)
			self.allele_keys.append(allele.allele_key)
			self.population.alleles_dictionary[allele.key] = allele
			
		
class Allele:
	'''
	A class for storing and manipulating information relating to a single allele.
	
	Key attributes:
	self.allele_key: (int) An integer identifying the allele among the other allels of the same locus.
	self.locus_key: (str) The single character identifying the locus of the allele.
	self.key: (str) The concatenation of the locus_key and the allele_key.
	self.locus: The Locus instance.
	self.initial_frequencies: The initial frequencies of the allele in each deme.
	self.male_indices/female_indices: The indices of male/female instances that has the allele. Homozygote indices are listed twice.
	'''
	def __init__(self, locus, allele_key, population):
		self.allele_key = allele_key
		self.locus_key = locus.key
		self.key = self.locus_key+str(self.allele_key)
		self.locus = locus
		self.population = population
		self.initial_frequencies = []
		self.male_indices = []
		self.female_indices = []
	
	def add_index(self, index, male):
		if male:
			self.male_indices.append(index)
		else:
			self.female_indices.append(index)
	
	def set_initial_frequency(self, frequency):
		self.initial_frequencies.append(frequency)
	
	def find_frequencies(self, male_frequencies, female_frequencies, print_to_screen = True):
		'''
		Find and return the frequency of the allele in each deme.
		'''
		if self.locus.microlinked:
			factor = 1.0/0.25
		elif self.locus.macrolinked:
			factor = 1.0/0.75
		else:
			factor = 1.0
		f = factor*0.25*(numpy.sum(male_frequencies[:,self.male_indices], axis = 1) + numpy.sum(female_frequencies[:,self.female_indices], axis = 1))
		if print_to_screen:
			print(self.key)
			print(f)
		return f

		
class Deme:
	'''
	A class for storing and manipulating information relating to a single deme.
	
	Key attributes:
	self.fitness_contributions: A dictionary of dictionaries, organized as {general_keys:{specific_keys: fitness_contribution}}, where a general_key
		gives only the locus/loci of the contribution (e.g. 'A' or 'A&B'), and a specific_key specifies the diplotype (e.g. 'A00' or 'A01&B01').
	self.preference_contributions: Same as self.fitness_contributions, but for preferences.
	self.static: (bool) If True, the deme is one the user expects to start in or quickly settle on an equilibrium, e.g. when the continent in a
		continent-island model experience no migration or evolution. Static demes are checked each generation for the first few generations if they are in
		equilibrium, and when they are, they are assumed to stay that way until the next full equilibrium (over all demes) is reached. This saves
		computional resources, as it allows the program to stop updating the genotype frequencies in the demes in question. Demes that are not static
		are called dynamic. NB! If you set deme.static = True for a deme that is not in, or close to, equilibrium, it will slow down the computation
		considerably!
	'''
	def __init__(self, key, index, population):
		self.key = key
		self.index = index
		self.population = population
		self.fitness_contributions = {}
		self.preference_contributions = {}
		self.static = False
		self.nonzero_indices = None
	
	def initialize_fitness_contributions(self):
		'''
		Loops over all entries in self.fitness_contributions, and assign the contribution to all genotype that correspond to the specific key.
		E.g. if specific key is 'A01&B01', then all genotypes having allele 0 and 1 for both locus A and locus B correspond.
		'''
		for general_key, specific_dictionary in self.fitness_contributions.items():
			for specific_key, fitness in specific_dictionary.items():
				diplotypes = set(specific_key.split('&'))
				for genotype in self.population.genotypes:
					if diplotypes.issubset(genotype.locus_diplotypes):
						genotype.add_fitness_contribution(self.index, general_key, fitness)
	
	def initialize_preference_contributions(self):
		'''
		Loops over all entries in self.preference_contributions, and assign the contribution to all couples that correspond to the specific key.
		E.g. if specific key is 'P00xT00', then all cuples for which the female is homozygous for the P0 allele and the male is homozygous for the T0 allele correspond.
		'''
		for general_key, specific_dictionary in self.preference_contributions.items():
			for specific_key, preference in specific_dictionary.items():
				female_diplotypes = set(specific_key.split('x')[0].split('&'))
				male_diplotypes = set(specific_key.split('x')[1].split('&'))
				for couple in self.population.couples:
					if female_diplotypes.issubset(couple.female.locus_diplotypes) and male_diplotypes.issubset(couple.male.locus_diplotypes):
						couple.add_preference_contribution(self.index, general_key, preference)


	def add_fitness_contribution(self, general_key, specific_key, fitness):
		if general_key in self.fitness_contributions:
			self.fitness_contributions[general_key][specific_key] = fitness
		else:
			self.fitness_contributions[general_key] = {specific_key: fitness}
	
	
	def add_preference_contribution(self, general_key, specific_key, preference_factor):
		if general_key in self.preference_contributions:
			self.preference_contributions[general_key][specific_key] = preference_factor
		else:
			self.preference_contributions[general_key] = {specific_key: preference_factor}
						
	
	def quickset_mating_preferences(self, info):
		'''
		A shortcut for implementng the prefernce parametrization used in the thesis. (Though note that this parametrization is not obligatory; any other parametrization
			can be implemented by using the add_preference_contribution_directly.)
		
		Arguments:
		info: a list of parameters:
			info[0]: key to the preference locus
			info[1]: key to the trait locus
			info[2]: beta_0, as defined in the thesis
			info[3]: beta_1, as defined in the thesis
			info[4]: tau, as defined in the thesis (optional, default is 1.0)
		'''
		l0 = info[0]
		l1 = info[1]
		if len(info) == 5:
			maximum = float(info[4])
		else:
			maximum = 1.0
		one_plus_a1 = 1 + float(info[2])
		one_plus_a2 = 1 + float(info[3])
		factor = maximum/max(one_plus_a1, one_plus_a2)
		general_key = l0 + 'x' + l1
		
		if not self.population.loci[l0].microlinked and not self.population.loci[l0].mitochondria and not self.population.loci[l1].microlinked and not self.population.loci[l1].mitochondria:
			self.add_preference_contribution(general_key, l0+'00x'+l1+'00', one_plus_a1*factor)
			self.add_preference_contribution(general_key, l0+'00x'+l1+'01', factor)
			self.add_preference_contribution(general_key, l0+'00x'+l1+'11', factor/one_plus_a1)
			self.add_preference_contribution(general_key, l0+'01x'+l1+'00', factor/( 0.5*( 1/one_plus_a1 + one_plus_a2 ) ))
			self.add_preference_contribution(general_key, l0+'01x'+l1+'01', factor)
			self.add_preference_contribution(general_key, l0+'01x'+l1+'11', factor/( 0.5*( 1/one_plus_a2 + one_plus_a1 ) ))
			self.add_preference_contribution(general_key, l0+'11x'+l1+'00', factor/one_plus_a2)
			self.add_preference_contribution(general_key, l0+'11x'+l1+'01', factor)
			self.add_preference_contribution(general_key, l0+'11x'+l1+'11', one_plus_a2*factor)
		
		self.add_preference_contribution(general_key, l0+'-0x'+l1+'00', one_plus_a1*factor)
		self.add_preference_contribution(general_key, l0+'-0x'+l1+'01', factor)
		self.add_preference_contribution(general_key, l0+'-0x'+l1+'11', factor/one_plus_a1)
		
		self.add_preference_contribution(general_key, l0+'-1x'+l1+'00', factor/one_plus_a2)
		self.add_preference_contribution(general_key, l0+'-1x'+l1+'01', factor)
		self.add_preference_contribution(general_key, l0+'-1x'+l1+'11', one_plus_a2*factor)
		
		self.add_preference_contribution(general_key, l0+'00x'+l1+'-0', one_plus_a1*factor)
		self.add_preference_contribution(general_key, l0+'01x'+l1+'-0', factor/( 0.5*( 1/one_plus_a1 + one_plus_a2 ) ))
		self.add_preference_contribution(general_key, l0+'11x'+l1+'-0', factor/one_plus_a2)
		
		self.add_preference_contribution(general_key, l0+'00x'+l1+'-1', factor/one_plus_a1)
		self.add_preference_contribution(general_key, l0+'01x'+l1+'-1', factor/( 0.5*( 1/one_plus_a2 + one_plus_a1 ) ))
		self.add_preference_contribution(general_key, l0+'11x'+l1+'-1', one_plus_a2*factor)
		
		self.add_preference_contribution(general_key, l0+'-0x'+l1+'-0', one_plus_a1*factor)
		self.add_preference_contribution(general_key, l0+'-0x'+l1+'-1', factor/one_plus_a1)
		self.add_preference_contribution(general_key, l0+'-1x'+l1+'-0', factor/one_plus_a2)
		self.add_preference_contribution(general_key, l0+'-1x'+l1+'-1', one_plus_a2*factor)
		
		
			
	def set_incompatibility(self, info):
		'''
		A shortcut for implementing the (four-allele) incompatibility parametrization used in the thesis.
		
		Arguments:
		info: a list of parameters:
			info[0]: Key to the first incompatibility locus
			info[1]: Key to the second incompatibility locus
			info[2]: S_E as defined in the thesis
			info[3]: h as defined in the thesis.
		'''
		
		l0 = info[0]
		l1 = info[1]
		s = float(info[2])
		h = float(info[3])
		general_key = l0 + '&' + l1
		if not self.population.loci[l0].microlinked and not self.population.loci[l0].mitochondria and not self.population.loci[l1].microlinked and not self.population.loci[l1].mitochondria:
			self.add_fitness_contribution(general_key, l0+'00&'+l1+'00', 1.0)
			self.add_fitness_contribution(general_key, l0+'11&'+l1+'11', 1.0)
			self.add_fitness_contribution(general_key, l0+'01&'+l1+'00', 1.0 - h*s)
			self.add_fitness_contribution(general_key, l0+'00&'+l1+'01', 1.0 - h*s)
			self.add_fitness_contribution(general_key, l0+'01&'+l1+'11', 1.0 - h*s)
			self.add_fitness_contribution(general_key, l0+'11&'+l1+'01', 1.0 - h*s)
			self.add_fitness_contribution(general_key, l0+'01&'+l1+'01', 1.0 - s)
			self.add_fitness_contribution(general_key, l0+'00&'+l1+'11', 1.0 - s)
			self.add_fitness_contribution(general_key, l0+'11&'+l1+'00', 1.0 - s)
			
		
		if (self.population.loci[l0].macrolinked or self.population.loci[l0].microlinked or self.population.loci[l0].mitochondria) and (self.population.loci[l0].macrolinked or self.population.loci[l0].microlinked or self.population.loci[l0].mitochondria):
			self.add_fitness_contribution(general_key, l0+'-0&'+l1+'-0', 1.0)
			self.add_fitness_contribution(general_key, l0+'-1&'+l1+'-1', 1.0)
			self.add_fitness_contribution(general_key, l0+'-0&'+l1+'-1', 1.0 - s)
			self.add_fitness_contribution(general_key, l0+'-1&'+l1+'-0', 1.0 - s)
			
			self.add_fitness_contribution(general_key, l0+'-0&'+l1+'00', 1.0)
			self.add_fitness_contribution(general_key, l0+'-0&'+l1+'01', 1.0 - h*s)
			self.add_fitness_contribution(general_key, l0+'-0&'+l1+'11', 1.0 - s)
			self.add_fitness_contribution(general_key, l0+'-1&'+l1+'00', 1.0 - s)
			self.add_fitness_contribution(general_key, l0+'-1&'+l1+'01', 1.0 - h*s)
			self.add_fitness_contribution(general_key, l0+'-1&'+l1+'11', 1.0)
			
			self.add_fitness_contribution(general_key, l0+'00&'+l1+'-0', 1.0)
			self.add_fitness_contribution(general_key, l0+'01&'+l1+'-0', 1.0 - h*s)
			self.add_fitness_contribution(general_key, l0+'11&'+l1+'-0', 1.0 - s)
			self.add_fitness_contribution(general_key, l0+'00&'+l1+'-1', 1.0 - s)
			self.add_fitness_contribution(general_key, l0+'01&'+l1+'-1', 1.0 - h*s)
			self.add_fitness_contribution(general_key, l0+'11&'+l1+'-1', 1.0)

class Equilibrium:
	'''
	A class for storing and manipulating information relating to a single equilibrium.
	
	Key attributes:
	self.key: (str) A string identifying the equilibrium (mostly used for communicating with the user)
	self.index: (int) All equilibria in a simulation are given an index indicating the order in which they occur
	self.action: (str) Indicates which action to be performed when the equilibrium condition is met.
		self.action = 'mutate': introduce mutation
		self.action = 'stop migration': stop migration
		self.action = 'start migration': start migration
		self.action = 'change migration': change migration matrix
		self.action = 'end': end simulation (this happens automatically for the last equilibrium)
	self.genotype_from/self.genotype_to: (str) The key to the genotype from/to which there is a mutation if self.action == 'mutate'
	self.frequency: (float) the frequency of the new mutation if self.action == 'mutate'
	self.condition_delta: If self.contidion_delta is not None, the equilibrium is reached and the equilibrium action is performed when all genotype frequencies
		change less than self.condition_delta (float) from one generation to the next
	self.condition_generation: If self.condition_generation is not Note, the equilibrium is reached and the quilibrium action is performed when there has been
		self.condition_generation (int) number of generations since the last equilibrium. If it is true that (self.condition_delta is not None) and (self.condition_genetation is not None),
		then then the equilibrium is reached whenever at least one of the conditions is reached.
	self.check: (int) Indicate how often the equilibrium conditions are to be checked. E.g. if (self.condition_delta is not None) and self.check == 100, then the program will compare
		the genotype frequencies in generation 100 and 101, 200 and 201, 300 and 301, etc.
	self.remove_dictionary: A dictionary giving the haplotypes and/or alleles to be removed from the simulation, e.g. because they never occur for this particular equilibrium
	
	'''
	def __init__(self, population, key, index, conditions_dictionary, action, check, condition_delta = None, condition_generation = None, remove_dictionary = {}, follow_dictionary = None, save_frequencies_filename = None, load_frequencies_filename = None, screen_tf = False, follow_filename = None):
		self.key = key
		self.index = index
		self.action = action
		self.population = population
		self.genotype_from = None
		self.genotype_to = None
		self.frequency = None
		self.deme_index = None
		self.compare = False
		self.check_frequency = check
		self.remove_dictionary = remove_dictionary
		self.migration = True
		self.condition_delta = condition_delta
		self.follow_dictionary = follow_dictionary
		self.save_frequencies_filename = save_frequencies_filename
		self.load_frequencies_filename = load_frequencies_filename
		
		self.new_migration_matrix = None

		if self.condition_delta is None:
			self.check_genotypes = False
		else:
			self.check_genotypes = True
		self.condition_generation = condition_generation
		if condition_generation is None:
			self.check_generation = False
		else:
			self.check_generation = True
		self.start_generation = None
		
		self.check_only_generation = False
		if not self.check_genotypes:
			if not self.check_generation:
				print('Error! No condition given for equilibrium ' + self.key + '!')
				sys.exit(1)
			else:
				self.check_only_generation = True
			
		
		self.save_follow_filename = follow_filename
		self.print_follow = screen_tf
		


		
		
	def initialize_follow(self):
		'''
		Initializes information about the alleles/haplotypes to be followed (i.e. those for which progress shall be printed or stored)
		'''
		self.follow_alleles = []
		self.follow_haplotypes = []
		
		if self.save_follow_filename is not None:
			self.save_follow = True
		else:
			self.save_follow = False
		
		if len(self.follow_dictionary['alleles']) > 0 or len(self.follow_dictionary['haplotypes']) > 0:
			self.follow = True
		else:
			self.follow = False
		
		for allele_key in self.follow_dictionary['alleles']:
			if allele_key in self.population.alleles_dictionary:
				self.follow_alleles.append(self.population.alleles_dictionary[allele_key])
			else:
				print('Error! Primary header # equilibrium, secondary header % follow: Allele key ' + allele_key + ' not found!')
				sys.exit(1)
				
		for haplotype_keys in self.follow_dictionary['haplotypes']:
			sub = Subtype(self.population, key = haplotype_keys)
			sub_haplotype_keys_set = set(haplotype_keys.split('&'))
			for haplotype in self.population.haplotypes:
				if sub_haplotype_keys_set.issubset(haplotype.locus_allele_keys):
					sub.add_indices(haplotype.male_indices, male = True)
					sub.add_indices(haplotype.female_indices, male = False)
			self.follow_haplotypes.append(sub)
		
		if len(self.follow_alleles)>0:

			self.follow_allele_frequencies = [[] for i in range(len(self.follow_alleles))]
			self.follow_alleles_bool = True
		else:
			self.follow_alleles_bool = False
		
		if len(self.follow_haplotypes)>0:
			self.follow_haplotype_frequencies = [[] for i in range(len(self.follow_haplotypes))]
			self.follow_haplotypes_bool = True
		else:
			self.follow_haplotypes_bool = False
			
			
		
		def add_indices(self, indices, male):
			if male:
				self.male_indices += indices
			else:
				self.female_indices += indices
	
	
	def finalize_info(self, reached_at_generation, allele_frequencies = None):
		'''
		FInalize information and store data files
		'''
		self.reached_at_generation = reached_at_generation
		if self.save_follow:
			if self.save_follow_filename == 'time':
				filename_root = time.strftime('%c').replace(' ','').replace(':','') + time.strftime('%S') + self.population.simulation.input_filename + '_Equilibrium' + self.key
			else:
				filename_root = self.save_follow_filename + '_Equilibrium' + self.key
			if self.follow_alleles_bool:
				for i in range(len(self.follow_alleles)):
					allele = self.follow_alleles[i]
					filename = filename_root + '_Allele_' + allele.key
					array = numpy.array(self.follow_allele_frequencies[i]).transpose()
					generations = numpy.linspace(0, self.reached_at_generation -self.start_generation, len(array[1]))
					numpy.save(filename, array)
					
			if self.follow_haplotypes_bool:
				for i in range(len(self.follow_haplotypes)):
					haplotype = self.follow_haplotypes[i]
					filename = filename_root + '_Haplotype_' + haplotype.key
					array = numpy.array(self.follow_haplotype_frequencies[i]).transpose()
					generations = numpy.linspace(0, self.reached_at_generation -self.start_generation, len(array[1]))
					numpy.save(filename, array)

	
	def check(self, male_frequencies_reduced, female_frequencies_reduced, dynamic_deme_indices, generation):
		'''
		Checks if at least one of the equilibrium conditions are reached
		
		Arguments:
		male_frequencies_reduced/female_frequencies reduced: A numpy array of the frequencies of all male/female frequencies
			except the ones that have been removed from consideration (because they never occur for this particular equilibrium)
		dynamic_deme_indices: The indices of the dynamic demes (see contructor)
		generation: (int) The current generation
		'''
		
		equilibrium_reached = False

		if self.follow and not self.compare:
			male_frequencies_full = copy.copy(self.population.male_zeros)
			female_frequencies_full = copy.copy(self.population.female_zeros)
		
			male_frequencies_full[:,self.male_keeper_indices] = male_frequencies_reduced # all male frequencies
			female_frequencies_full[:,self.female_keeper_indices] = female_frequencies_reduced # all female frequencies
			
			# find the frequnecies of the alleles and haplotypes that are followed:
			if self.follow_alleles_bool: 
				for i in range(len(self.follow_alleles)):
					self.follow_allele_frequencies[i].append(self.follow_alleles[i].find_frequencies(male_frequencies_full, female_frequencies_full, print_to_screen = self.print_follow))
			if self.follow_haplotypes_bool:
				for i in range(len(self.follow_haplotypes)):
					self.follow_haplotype_frequencies[i].append(self.follow_haplotypes[i].find_frequencies(male_frequencies_full, female_frequencies_full, print_to_screen = self.print_follow))
		
		if self.condition_delta is not None: # check if all genotype frequencies have changed less than self.condition_delta from the previous generation
			if self.compare:
				male_max_change = numpy.max(abs(self.old_male_frequencies_reduced-male_frequencies_reduced))
				female_max_change = numpy.max(abs(self.old_female_frequencies_reduced-female_frequencies_reduced))
				print('Equilibrium ' + self.key)
				print('Male max change:', male_max_change)
				print('Female max change:', female_max_change)
				if max(male_max_change, female_max_change) < self.condition_delta:
					equilibrium_reached = True
				else:
					self.old_male_frequencies_reduced = male_frequencies_reduced
					self.old_female_frequencies_reduced = female_frequencies_reduced
			
			else:
				self.old_male_frequencies_reduced = male_frequencies_reduced
				self.old_female_frequencies_reduced = female_frequencies_reduced
				
				
			
			if equilibrium_reached:
				self.finalize_info(generation)
		
		if (not equilibrium_reached) and (self.condition_generation is not None): # check if the total number of generations since the previous equilibrium is more than the (optional) maximum
			if generation-self.start_generation > self.condition_generation:
				self.finalize_info(generation)
				equilibrium_reached = True
			else:
				equilibrium_reached = False
		
		self.compare = not self.compare
		return equilibrium_reached
		
				
	
	def execute_action(self, male_frequencies = None, female_frequencies = None):
		'''
		Executes the equilibrium action if self.action == 'mutate' (other equilibria actions are handled elsewhere)
		'''
		if self.action == 'mutate': # introduce new mutation
			if self.mutate_male:
				if male_frequencies[self.deme_index][self.male_from_index] < self.frequency:
					print('Error! Introduction frequency for the new mutant at equilibrium ' + self.key + ' is larger than the frequency of the genotype from which it is supposed to mutate!')
					sys.exit(1)
				male_frequencies[self.deme_index][self.male_from_index] -= self.frequency
				male_frequencies[self.deme_index][self.male_to_index] += self.frequency
			if self.mutate_female:
				if female_frequencies[self.deme_index][self.female_from_index] < self.frequency:
					print('Error! Introduction frequency for the new mutant at equilibrium ' + self.key + ' is larger than the frequency of the genotype from which it is supposed to mutate!')
					sys.exit(1)
				female_frequencies[self.deme_index][self.female_from_index] -= self.frequency
				female_frequencies[self.deme_index][self.female_to_index] += self.frequency
			
			return (male_frequencies, female_frequencies)
	
	def find_keepers(self):
		'''
		Find the remaining indices after the genotypes to be removed are removed.
		'''
		male_keepers = numpy.array([True for i in range(self.population.males_n)])
		female_keepers = numpy.array([True for i in range(self.population.females_n)])
		males = self.population.males
		females = self.population.females
		for allele_key in self.remove_dictionary['allele_keys']:
			remove_allele = self.population.alleles_dictionary[allele_key]
			male_keepers[remove_allele.male_indices] = False
			female_keepers[remove_allele.female_indices] = False
	
		for sub_haplotype_key in self.remove_dictionary['haplotype_keys']:
			sub_haplotype_keys_set = set(sub_haplotype_key.split('&'))
			for haplotype in self.population.haplotypes:
				if sub_haplotype_keys_set.issubset(haplotype.locus_allele_keys):
					male_keepers[haplotype.male_indices] = False
					female_keepers[haplotype.female_indices] = False
				
		self.male_keeper_tf = male_keepers # A list of boolean variables indicating whether each male genotype shall be kept
		self.female_keeper_tf = female_keepers
	
		self.male_keeper_indices = numpy.where(male_keepers)[0] # indices for genotypes that shall be kept (= not removed)
		self.female_keeper_indices = numpy.where(female_keepers)[0]
	
		self.male_remove_indices = numpy.where(self.male_keeper_tf == False)[0] # indices for genotypes that shall be removed
		self.female_remove_indices = numpy.where(self.female_keeper_tf == False)[0] 
		
		self.couple_remove_indices = numpy.where(numpy.outer(self.male_keeper_tf, self.female_keeper_tf).ravel() == False)[0] # indices for couples that shall be removed

	
	def find_reduced_tensors(self):
		'''
		Finds the reduced fitness, preference, and offspring tensors (i.e. the tensors after the genotypes to be removed are removed)
		'''
		demes_n = self.population.demes_n
	
		male_fitness_full = self.population.male_fitness_array
		female_fitness_full = self.population.female_fitness_array
	
		self.male_fitness_reduced = numpy.array([male_fitness_full[d][self.male_keeper_indices] for d in range(demes_n)])
		self.female_fitness_reduced = numpy.array([female_fitness_full[d][self.female_keeper_indices] for d in range(demes_n)])
	
	
		preference_tensor_full = self.population.preference_tensor
	
		preference_tensor_reduced = numpy.array([numpy.delete(preference_tensor_full[d], self.male_remove_indices, axis = 0) for d in range(demes_n)])
		self.preference_tensor_reduced = numpy.array([numpy.delete(preference_tensor_reduced[d], self.female_remove_indices, axis = 1) for d in range(demes_n)])
		male_offspring_tensor_full = self.population.male_offspring_tensor
	
		male_offspring_tensor_reduced = numpy.delete(male_offspring_tensor_full, self.couple_remove_indices, axis = 0)
		self.male_offspring_tensor_reduced = numpy.delete(male_offspring_tensor_reduced, self.male_remove_indices, axis = 1)
		
		female_offspring_tensor_full = self.population.female_offspring_tensor
	
		female_offspring_tensor_reduced = numpy.delete(female_offspring_tensor_full, self.couple_remove_indices, axis = 0)
		self.female_offspring_tensor_reduced = numpy.delete(female_offspring_tensor_reduced, self.female_remove_indices, axis = 1)
	
		self.male_keys_reduced = list(numpy.array(self.population.male_keys)[self.male_keeper_indices])
		self.female_keys_reduced = list(numpy.array(self.population.female_keys)[self.female_keeper_indices])
		
	
	def initialize_action(self):
		'''
		Initializes the equilibrium action (if the action is to introduce a new mutation)
		'''
		if self.action == 'mutate':
			genotype_from = self.genotype_from # genotype to mutate from
			genotype_to = self.genotype_to # genotype to mutate to
			# sex determination diplotype for males/females:
			male_key = self.population.male_key 
			female_key = self.population.female_key
			
			if male_key in genotype_from:
				self.mutate_male = True
				self.mutate_female = False
				genotype_from = genotype_from.replace(male_key, '')
				genotype_to = genotype_to.replace(male_key, '')
			elif female_key in genotype_from:
				self.mutate_male = False
				self.mutate_female = True
				genotype_from = genotype_from.replace(female_key, '')
				genotype_to = genotype_to.replace(female_key, '')
				
			else: # mutate both male and female unless otherwise specified
				self.mutate_male = True
				self.mutate_female = True
			
			# find indices of from/to genotypes
			if self.mutate_male:
				self.male_from_index = self.population.male_keys.index(genotype_from)
				self.male_to_index = self.population.male_keys.index(genotype_to)
			
			if self.mutate_female:
				self.female_from_index = self.population.female_keys.index(genotype_from)
				self.female_to_index = self.population.female_keys.index(genotype_to)
				
		
	
class Simulation:
	'''
	A class for storing and manipulating information relating to a single simulation
	
	Key attributes:
	self.equilibria: A list of Equilibria instances, sorted in the order in which they occur in the simulation
	self.population: A pointer to the Population instance
	'''
	def __init__(self):
		self.population = Population(self)
		self.calculator = Calculator(self.population)
		self.population.add_calculator(self.calculator)
		self.equilibria = []
		self.report = None
	
	def initialize_equilibrium_follow(self):
		for equilibrium in self.equilibria:
			equilibrium.initialize_follow()
	
	def find_equilibria_keepers(self):
		for equilibrium in self.equilibria:
			equilibrium.find_keepers()
	
	def initialize_equilibria(self):
		print('Initializing equilibria...')
		for equilibrium in self.equilibria:
			equilibrium.initialize_action()
			equilibrium.find_keepers()
			equilibrium.find_reduced_tensors()
	
	
	def add_equilibrium(self, equilibrium):
		self.equilibria.append(equilibrium)
		
	def read_input(self, input_filename):
		'''
		Reads input from the input file and initializes the population
		
		Arguments:
		input_filename: (str) The name of or path to the input file
		'''
		print('*'+input_filename+'*')
		self.input_filename = input_filename
		input_file = open(input_filename)
		input_text = input_file.read()
		input_file.close()
		
		# Read chromosome input
		chromosome_pattern = '#\s*C*c*hromosome\s(?P<key>[A-Za-z\d]+)(?P<info>[^#]+)'
		chromosome_matches = re.finditer(chromosome_pattern, input_text)
		chromosome_index = 0
		for chromosome_match in chromosome_matches:
			chromosome_key = chromosome_match.group('key').strip()
			print('Reads input for chromosome ' + chromosome_key + '...')
			chromosome_info = chromosome_match.group('info')
			loci_pattern = '[Ll]oci\s*=\s*(?P<loci_keys>[,@\$A-Z\[\]]+)'
			lambda_pattern = '[Ll]ambda\s*=\s*(?P<lambda_values>[\d,.\s]+)'
			mu_pattern = '[Mm]u\s*=\s*(?P<mu_values>[\d,.\s]+)'
			d_pattern = '^d\s*=\s*(?P<d_values>[\d,.\s]+)'		
			alleles_ns_pattern = '%\s*(A|a)lleles(?P<info>[^%#]+)'
			loci_match = re.search(loci_pattern, chromosome_info)
			alleles_ns_match = re.search(alleles_ns_pattern, chromosome_info)
			lambda_match = re.search(lambda_pattern, chromosome_info)
			mu_match = re.search(mu_pattern,chromosome_info)
			d_match = re.search(d_pattern, chromosome_info, re.MULTILINE)
					
			if loci_match:
				loci_keys = loci_match.group('loci_keys').strip().replace(',','')
			else:
				print('Error! Loci keys for chromosome '+ chromosome_key + ' not found in input file!')
				
			allele_ns_dictionary = {}
			if alleles_ns_match:
				alleles_ns_info = alleles_ns_match.group('info')
				allele_ns_pattern = '^(?P<key>[\[\]@A-Z]+)\s*=\s*(?P<n>[\d]+$)'
				allele_ns_matches = re.finditer(allele_ns_pattern, alleles_ns_info, re.MULTILINE)
				for allele_ns_match in allele_ns_matches:
					locus_key = allele_ns_match.group('key').strip().replace(' ','')
					n = int(allele_ns_match.group('n').strip().replace(' ',''))
					allele_ns_dictionary[locus_key] = n
				
			else:
				print('NB! Allele numbers for chromosome '+ chromosome_key + ' not found in input file! All allele numbers set to 2 by default.')
				allele_ns = []
				for l in loci_keys:
					if l != '$' and l != '[' and l != ']':
						if l != '@':
							allele_ns.append(2)
						else:
							allele_ns.append(1)
			if lambda_match:
				lambda_values = list(map(float,lambda_match.group('lambda_values').strip().replace(' ', '').split(',')))
			else:
				print('NB! No input found for Lambda in input file. All lambda values are set to 0 by default')
				lambda_values = [0.0 for i in range(len(loci_keys)-1)]
			if mu_match:
				mu_values = list(map(float,mu_match.group('mu_values').strip().replace(' ', '').split(',')))
			else:
				print('NB! No input found for Mu in input file. All mu values are set to 0 by default')
				mu_values = [0.0 for i in range(len(loci_keys)-1)]
			
			if d_match:
				d_values = list(map(float,d_match.group('d_values').strip().replace(' ', '').split(',')))
			else:
				print('NB! No input found for d in input file. All d values are set to 1 by default')
				d_values = [1.0 for i in range(len(loci_keys)-1)]
			
			if '$' in loci_keys: # sex determination locus
				macrolinked_pattern = '[Mm]acro-*linked\s*=\s*(?P<macrolinked>[@A-Z,\s]+)$'
				microlinked_pattern = '[Mm]icro-*linked\s*=\s*(?P<microlinked>[@A-Z,\s]+)$'
				heterogametic_pattern = '[Hh]etero-*gametic\s*=\s*(?P<heterogametic>[A-Za-z]+)$'			
				macrolinked_match = re.search(macrolinked_pattern, chromosome_info, re.MULTILINE)
				microlinked_match = re.search(microlinked_pattern, chromosome_info, re.MULTILINE)
				heterogametic_match = re.search(heterogametic_pattern, chromosome_info, re.MULTILINE)
				microlinked_loci = []
				macrolinked_loci = []
				
				if macrolinked_match:
					macrolinked_loci = macrolinked_match.group('macrolinked').strip().replace(' ','').split(',')
				if microlinked_match:
					microlinked_loci = microlinked_match.group('microlinked').strip().replace(' ','').split(',')
				if heterogametic_match:
					male_matches = ['M', 'm', 'Male', 'male']
					female_matches = ['F', 'f', 'Female', 'female']
					
					if heterogametic_match.group('heterogametic') in male_matches:
						male_heterogametic = True
					elif heterogametic_match.group('heterogametic') in female_matches:
						male_heterogametic = False
					else:
						print('Error! Invalid input for heterogametic sex. Write either Heterogametic = Male or Heterogametic = Female')
				
				sex_chromosome = Sex_chromosome(population = self.population, key = chromosome_key, loci_keys = loci_keys, allele_ns_dictionary = allele_ns_dictionary, lambda_values = lambda_values, mu_values = mu_values, d_values = d_values, sex = True, index = chromosome_index, macrolinked_loci = macrolinked_loci, microlinked_loci = microlinked_loci, male_heterogametic = male_heterogametic)
				self.population.sex_chromosome = sex_chromosome
				self.population.sex_chromosome_index = chromosome_index
				sex_chromosome.initialize_loci()
				self.population.add_chromosome(sex_chromosome)
			
			else:
				chromosome = Chromosome(population = self.population, key = chromosome_key, loci_keys = loci_keys, allele_ns_dictionary = allele_ns_dictionary, lambda_values = lambda_values, mu_values = mu_values, d_values = d_values, sex = False, index = chromosome_index)
				chromosome.initialize_loci()
				self.population.add_chromosome(chromosome)
			
			chromosome_index += 1
			
		# Read deme input:
		deme_pattern = '#\s*[Dd]eme\s(?P<key>[A-Za-z\d]+)(?P<info>[^#]+)'
		deme_matches = re.finditer(deme_pattern, input_text)
		deme_index = 0
		male_genotype_frequencies = []
		female_genotype_frequencies = []
		for deme_match in deme_matches:
			deme_key = deme_match.group('key').strip()
			deme_info = deme_match.group('info')
			deme = Deme(key = deme_key, index = deme_index, population = self.population)
			self.population.demes.append(deme)
			self.population.deme_keys.append(deme_key)
			static_pattern = '^\s*[Ss]tatic\s*=\s*(?P<static_tf>[A-Za-z]+)\s*$'
			static_match = re.search(static_pattern, deme_info, re.MULTILINE)
			if static_match:
				static_tf = static_match.group('static_tf').replace(' ','').lower()
				if static_tf == 'true' or static_tf == 't':
					deme.static = True
				elif static_tf == 'false' or static_tf == 'f':
					deme.static = False
				else:
					print('NB! Invalid input for static in section deme ' + deme_key +'. Deme is set to non-static by default')
					deme.static = False
			else:
				deme.static = False
			
			if deme.static:
				self.population.static_deme_indices.append(deme_index)
			else:
				self.population.dynamic_deme_indices.append(deme_index)
			
			allele_frequencies_pattern = '%\s*(A|a)llele\s(F|f)requenc(ies|y)\n(?P<info>[^%#]+)'
			allele_frequencies_match = re.search(allele_frequencies_pattern, deme_info)
			if allele_frequencies_match:
				allele_frequencies_info = allele_frequencies_match.group('info')
				allele_frequency_pattern = '^(?P<key>[\[\]@?\dA-Za-z]+)\s*=\s*(?P<frequency>[\d,.]+$)'
				allele_frequency_matches = re.finditer(allele_frequency_pattern, allele_frequencies_info, re.MULTILINE)
				
				multiple_allele_frequencies_pattern = '^(?P<keys>[\[\]@?\dA-Za-z\{\}\|]+)\s*=\s*(?P<frequency>[\d,.]+$)'
				multiple_allele_frequencies_matches = re.finditer(multiple_allele_frequencies_pattern, allele_frequencies_info, re.MULTILINE)
				allele_frequency_dictionary = {}
				for multiple_allele_frequencies_match in multiple_allele_frequencies_matches:
					keys = multiple_allele_frequencies_match.group('keys').strip().replace(' ','')
					try:
						frequency = float(multiple_allele_frequencies_match.group('frequency').strip().replace(',','.'))
					except:
						print('Error! Invalid frequency given for alleles ' + keys + ' in deme ' + deme_key + '!')
						sys.exit(1)
					
					keys = keys.replace('{','').replace('}','')
					allele_key = keys[-1]
					locus_keys = keys[:-1].split('|')
					for locus_key in locus_keys:
						key = locus_key+allele_key
						allele_frequency_dictionary[key] = frequency
				allele_frequency_dictionary['?-'] = 1.0
				for chromosome in self.population.chromosomes:
					for locus in chromosome.loci:
						for allele in locus.alleles:
							if allele.locus.sex:
								if allele.allele_key == 1:
									allele.set_initial_frequency(1.0)
								else:
									allele.set_initial_frequency(0.5)
							elif chromosome.sex and chromosome.heteromorphic and allele.locus.breakpoint:
								allele.set_initial_frequency(1.0)
							elif allele.key in allele_frequency_dictionary:
								allele.set_initial_frequency(float(allele_frequency_dictionary[allele.key]))
							elif '?'+str(allele.allele_key) in allele_frequency_dictionary:
								allele.set_initial_frequency(float(allele_frequency_dictionary['?'+str(allele.allele_key)]))
			
			genotype_frequencies_pattern = '%\s*(G|g)enotype\s(F|f)requenc(ies|y)\n(?P<info>[^%#]+)'
			genotype_frequencies_match = re.search(genotype_frequencies_pattern, deme_info)
			male_genotype_frequencies_dictionary = {}
			female_genotype_frequencies_dictionary = {}
			if genotype_frequencies_match:
				genotype_frequencies_info = genotype_frequencies_match.group('info')
				genotype_frequency_pattern = '^(?P<key>[@\dA-Z$\-\[\]]+)\s*=\s*(?P<frequency>[\d,.e\-\+]+$)'
				genotype_frequency_matches = re.finditer(genotype_frequency_pattern, genotype_frequencies_info, re.MULTILINE)
				for genotype_frequency_match in genotype_frequency_matches:
					genotype_key = genotype_frequency_match.group('key').strip()
					genotype_frequency = float(genotype_frequency_match.group('frequency').strip().replace(',','.'))
					if '$01' in genotype_key:
						if self.population.male_heterogametic:
							male_genotype_frequencies_dictionary[genotype_key.replace('$01','').replace('$10','')] = genotype_frequency
						else:
							female_genotype_frequencies_dictionary[genotype_key.replace('$01','').replace('$10','')] = genotype_frequency
					elif '$11' in genotype_key:
						if self.population.male_heterogametic:
							female_genotype_frequencies_dictionary[genotype_key.replace('$11','')] = genotype_frequency
						else:
							male_genotype_frequencies_dictionary[genotype_key.replace('$11','')] = genotype_frequency
					else:
						male_genotype_frequencies_dictionary[genotype_key] = genotype_frequency
						female_genotype_frequencies_dictionary[genotype_key] = genotype_frequency
						

			male_genotype_frequencies.append(male_genotype_frequencies_dictionary)
			female_genotype_frequencies.append(female_genotype_frequencies_dictionary)
			
			fitness_pattern = '^%\s*[Ff]itness(?P<info>[^#%]+)'
			fitness_match = re.search(fitness_pattern, deme_info, re.MULTILINE)
			if fitness_match:
				fitness_info = fitness_match.group('info')
				self.add_fitness_contributions(fitness_info, deme)
			
			mating_pattern = '^%\s*[Mm]ating(?P<info>[^#%]+)'
			mating_match = re.search(mating_pattern, deme_info, re.MULTILINE)
			if mating_match:
				mating_info = mating_match.group('info')
				self.add_preference_contributions(mating_info, deme)
			
			
			deme_index += 1
		
		# Read population input
		self.population.set_demes_n()
		population_pattern = '^#\s*[Pp]opulation(?P<info>[^#]+)'
		population_match = re.search(population_pattern, input_text, re.MULTILINE)
		if population_match:
			population_info = population_match.group('info')
			gamma_pattern = '^\s*[Gg]amma\s*=\s*(?P<gamma>[\d,.\s]+)'
			alpha_pattern = '^\s*[Aa]lpha\s*=\s*(?P<tf>[A-Za-z]+)$'
			c_pattern = '^\s*[Cc]\s*=\s*(?P<c>[\d.\se\-\+]+)$'
			c_match = re.search(c_pattern, population_info, re.MULTILINE)
			gamma_match = re.search(gamma_pattern, population_info, re.MULTILINE)
			remove_pattern = '%\s*[Rr]emove\s*\n(?P<info>[^%#]+)'
			remove_match = re.search(remove_pattern, population_info, re.MULTILINE)
			alpha_match = re.search(alpha_pattern, population_info, re.MULTILINE)
			
			save_offspring_pattern = '^\s*[Ss]ave_offspring\s*=\s*(?P<filename>[^\n]+)$'
			load_offspring_pattern = '^\s*[Ll]oad_offspring\s*=\s*(?P<filename>[^\n]+)$'
			
			save_offspring_match = re.search(save_offspring_pattern, population_info, re.MULTILINE)
			load_offspring_match = re.search(load_offspring_pattern, population_info, re.MULTILINE)
			alpha = True
			if alpha_match:
				alpha_str = alpha_match.group('tf').strip().lower()
				if alpha_str == 't' or alpha_str == 'true':
					alpha = True
				elif alpha_str == 'f' or alpha_str == 'false':
					alpha = False
				else:

					print('Error! Invalid input parameter alpha under header # population! Please revise')
					sys.exit(1)
			
			self.population.alpha = alpha
			
			if save_offspring_match:
				save_offspring_filename = save_offspring_match.group('filename').strip()
				self.population.save_offspring_filename = save_offspring_filename

			if load_offspring_match:
				load_offspring_filename = load_offspring_match.group('filename').strip()
				self.population.load_offspring_filename = load_offspring_filename
			
			if c_match:
				c = float(c_match.group('c').strip().replace(' ',''))
				self.population.c = c
			else:
				print('NB! No input found for c. Using c=0 by default')
			if remove_match:
				remove_info = remove_match.group('info')
				haplotypes_pattern = '^\s*[Hh]aplotypes*\s*=\s*(?P<haplotype_keys>[A-Z\d,@\$\s&\[\]]+)$'
				haplotypes_match = re.search(haplotypes_pattern, remove_info)
				if haplotypes_match:
					haplotypes_to_remove = haplotypes_match.group('haplotype_keys').replace(' ','').strip().split(',')
				else:
					haplotypes_to_remove = []
				print('Haplotypes to remove:', haplotypes_to_remove)
				self.population.haplotypes_to_remove = haplotypes_to_remove
				
			if gamma_match:
				gamma_values = gamma_match.group('gamma').replace(' ', '')
				gamma_values = gamma_values.split(',')
				gamma_values = list(map(float,gamma_values))
				self.population.set_gamma_values(gamma_values)
			
			fitness_pattern = '^%\s*[Ff]itness(?P<info>[^#%]+)'
			fitness_match = re.search(fitness_pattern, population_info, re.MULTILINE)
			if fitness_match: 
				fitness_info = fitness_match.group('info')
				for deme in self.population.demes:
					self.add_fitness_contributions(fitness_info, deme)
			
			mating_pattern = '^%\s*[Mm]ating(?P<info>[^#%]+)'
			mating_match = re.search(mating_pattern, population_info, re.MULTILINE)
			if mating_match:
				mating_info = mating_match.group('info')
				for deme in self.population.demes:
					self.add_preference_contributions(mating_info, deme)
			
			interaction_pattern = '^%\s*[Ii]nteractions*(?P<info>[^#%]+)'
			interaction_match = re.search(interaction_pattern, population_info, re.MULTILINE)
			if interaction_match:
				interaction_info = interaction_match.group('info')
				fitness_pattern = '^\s*[Ff]itness\s*=\s*(?P<fitness_interaction>[$@&A-Z*\-\/+()\[\]]+)'
				fitness_match = re.search(fitness_pattern, interaction_info, re.MULTILINE)
				if fitness_match:
					fitness_interaction = fitness_match.group('fitness_interaction').replace(' ','').strip()
					self.population.fitness_interaction = fitness_interaction
				preference_pattern = '^\s*[Pp]reference\s*=\s*(?P<preference_interaction>[$@&A-Z*\-\/+()x\[\]]+)'
				preference_match = re.search(preference_pattern, interaction_info, re.MULTILINE)
				if preference_match:
					preference_interaction = preference_match.group('preference_interaction').replace(' ','').strip()
					self.population.preference_interaction = preference_interaction
			migration_pattern = '^%\s*[Mm]igration(?P<matrix>[\d.\s\n]+)'
			migration_match = re.search(migration_pattern, input_text, re.MULTILINE)
			if migration_match:
				matrix_info = migration_match.group('matrix').strip().split('\n')
				matrix = []
				for r in matrix_info:
					row = r.split(' ')
					row = list(map(float,row))
					matrix.append(row)
				matrix = numpy.array(matrix)

				self.population.set_migration_matrix('default', matrix)
				
		
		# Read equilibrium input
		equilibrium_pattern = '^#\s*[Ee]quilibrium\s(?P<key>[A-Za-z\d]+)(?P<info>[^#]+)'
		equilibrium_matches = re.finditer(equilibrium_pattern, input_text, re.MULTILINE)
		index = 0
		for equilibrium_match in equilibrium_matches:
			conditions_dictionary = {}
			remove_dictionary = {'allele_keys': [], 'haplotype_keys': []}
			equilibrium_info = equilibrium_match.group('info')
			equilibrium_key = equilibrium_match.group('key').strip()
			conditions_pattern = '^%\s*[Cc]onditions*\s*\n(?P<info>[^#%]+)'
			conditions_match = re.search(conditions_pattern, equilibrium_info, re.MULTILINE)
			check_pattern = '^\s*[Cc]heck\s*=\s*(?P<check_frequency>[\d]+)$'
			check_match = re.search(check_pattern, equilibrium_info, re.MULTILINE)
			remove_pattern = '%\s*[Rr]emove\s*\n(?P<info>[^%#]+)'
			remove_match = re.search(remove_pattern, equilibrium_info)
			
			follow_pattern = '%\s*[Ff]ollow(?P<info>[^#%]+)'
			follow_match = re.search(follow_pattern, equilibrium_info)
			
			save_pattern = '^\s*[Ss]ave\s*=\s*(?P<filename>[^\n]+)$'
			save_match = re.search(save_pattern, equilibrium_info, re.MULTILINE)
			
			load_pattern = '^\s*[Ll]oad\s*=\s*(?P<filename>[^\n]+)$'
			load_match = re.search(load_pattern, equilibrium_info, re.MULTILINE)
			
			if save_match:
				save_frequencies_filename = save_match.group('filename').strip()
			
			else:
				save_frequencies_filename = None
			
			if load_match:
				load_frequencies_filename = load_match.group('filename').strip()
			
			else:
				load_frequencies_filename = None
				
			follow_dictionary = {'genotypes':[], 'haplotypes' : [], 'alleles' : []}
			screen_tf = False
			follow_filename = None
			
			if follow_match:
				follow_info = follow_match.group('info')
				alleles_pattern = '^\s*[Aa]lleles*\s*=\s*(?P<allele_keys>[A-Za-z\d,@$\s\[\]]+)$'
				haplotypes_pattern = '^\s*[Hh]aplotypes*\s*=\s*(?P<haplotype_keys>[A-Z\d,@$\s&\[\]]+)$'
				genotypes_pattern = '^\s*[Gg]enotypes\s*=\s*(?P<genotypes>[A-Za-z\d,@$\s\[\]]+)$'
				screen_pattern = '^\s*[Ss]creen\s*=\s*(?P<tf>[A-Za-z]+)$'
				follow_file_pattern = '^\s*[Ff]ile\s*=\s*(?P<filename>[^\n]+)$'
			
				alleles_match = re.search(alleles_pattern, follow_info, re.MULTILINE)
				haplotypes_match = re.search(haplotypes_pattern, follow_info, re.MULTILINE)
				genotypes_match = re.search(genotypes_pattern, follow_info, re.MULTILINE)
				screen_match = re.search(screen_pattern, follow_info, re.MULTILINE)
				follow_file_match = re.search(follow_file_pattern, follow_info, re.MULTILINE)
				
				if screen_match:
					screen_tf_str = screen_match.group('tf').strip().replace(' ','').lower()
					if screen_tf_str == 't' or screen_tf_str == 'true':
						screen_tf = True
					else:
						screen_tf = False
				else:
					screen_tf = False
				
				if follow_file_match:
					follow_filename = follow_file_match.group('filename')
				else:
					follow_filename = None
				
				if genotypes_match:
					if genotypes_match.group('genotypes').replace(' ','').lower() == 'all':
						genotype_keys = 'all'
					else:
						genotype_keys = genotypes_match.group('genotypes').replace(' ','').strip().split(',')
					follow_dictionary['genotypes'] = genotype_keys
				else:
					genotype_keys = None

			
				if alleles_match:
					allele_keys = alleles_match.group('allele_keys').replace(' ','').strip().split(',')
					follow_dictionary['alleles'] = allele_keys
			
				if haplotypes_match:
					haplotype_keys = haplotypes_match.group('haplotype_keys').replace(' ','').strip().split(',')
					follow_dictionary['haplotypes'] = haplotype_keys
				else:
					haplotype_keys = []
				
			
			if remove_match:
				remove_info = remove_match.group('info')
				alleles_pattern = '^\s*[Aa]lleles*\s*=\s*(?P<allele_keys>[A-Za-z\d,@$\s\[\]]+)$'
				haplotypes_pattern = '^\s*[Hh]aplotypes*\s*=\s*(?P<haplotype_keys>[A-Z\d,@$\s&\[\]]+)$'
				genotypes_pattern = '^\s*[Gg]enotypes\s*=\s*(?P<genotypes>[A-Za-z\d,@$\s\[\]]+)$'
				genotypes_match = re.search(genotypes_pattern, remove_info, re.MULTILINE)
				alleles_match = re.search(alleles_pattern, remove_info, re.MULTILINE)
				haplotypes_match = re.search(haplotypes_pattern, remove_info, re.MULTILINE)
				if genotypes_match:
					if genotypes_match.group('genotypes').replace(' ','').lower() == 'all':
						genotype_keys = 'all'
					else:
						genotype_keys = genotypes_match.group('genotypes').replace(' ','').strip().split(',')
				else:
					genotype_keys = []
				remove_dictionary['genotype_keys'] = genotype_keys

			
				if alleles_match:
					allele_keys = alleles_match.group('allele_keys').replace(' ','').strip().split(',')
				else:
					allele_keys = []
				
				
				remove_dictionary['allele_keys'] = allele_keys
				
				if haplotypes_match:
					haplotype_keys = haplotypes_match.group('haplotype_keys').replace(' ','').strip().split(',')
				else:
					haplotype_keys = []
					
				remove_dictionary['haplotype_keys'] = haplotype_keys
				
				
			if check_match:
				try:
					check = int(check_match.group('check_frequency'))
				except:
					print('Error! Input for parameter check under header #Equilibrium ' + equilibrium_key + ' is not a valid digit! Please revise.')
					sys.exit(1)
			else:
				print('NB! No input found for parameter check under header #Equilibrium ' + equilibrium_key +'. Will check every 500th generation by default')
				check = 500
			if conditions_match:
				conditions_info = conditions_match.group('info').strip()
				delta_pattern = '^[Dd]elta\s*[<=]\s*(?P<delta_frequency>[\d.e\+\-]+)$'
				generation_pattern = '^[Gg]enerations*\s*=\s*(?P<generation_number>[\d.e\+]+)$'
				delta_match = re.search(delta_pattern, conditions_info, re.MULTILINE)
				generation_match = re.search(generation_pattern, conditions_info, re.MULTILINE)
				if delta_match:
					try:
						condition_delta = float(delta_match.group('delta_frequency').strip().replace(' ',''))
					except:
						print('Error! Invalid input for parameter delta under header #Equilibrium ' + equilibrium_key + '! Please revise.')
						sys.exit(1)
				else:
					condition_delta = None
				if generation_match:
					try:
						condition_generation = int(float(generation_match.group('generation_number').strip().replace(' ','')))
					except:
						print('Error! Invalid input for parameter generation under header #Equilibrium ' + equilibrium_key + '! Please revise.')
						sys.exit(1)
				else:
					condition_generation = None
				
			else:
				print('Error! No conditions given for #Equilibrium ', equilibrium_key, '!')
				sys.exit(1)
			
		
				
			
			action_pattern = '^%\s*[Dd]o\s(?P<do_what>[A-Za-z ]+)(?P<info>[^#]+)'
			action_match = re.search(action_pattern, equilibrium_info, re.MULTILINE)
			if action_match:
				action = action_match.group('do_what').strip().lower()
				equilibrium = Equilibrium(self.population, equilibrium_key, index, conditions_dictionary, action, check, condition_delta = condition_delta, condition_generation = condition_generation, remove_dictionary = remove_dictionary, save_frequencies_filename = save_frequencies_filename, load_frequencies_filename = load_frequencies_filename, follow_dictionary = follow_dictionary, screen_tf = screen_tf, follow_filename = follow_filename)
				self.add_equilibrium(equilibrium)
				action_info = action_match.group('info')
				if action == 'change migration':
					matrix_info = action_info.strip().split('\n')
					matrix = []
					for r in matrix_info:
						row = r.split(' ')
						row = list(map(float,row))
						matrix.append(row)
					matrix = numpy.array(matrix)
					equilibrium.new_migration_matrix = matrix
				
				if action == 'mutate':
					genotype_pattern = '^(?P<from>[A-Z\d@\[\]$\-]+)\sto\s(?P<to>[A-Z\d@\[\]$\-]+$)'
					genotype_match = re.search(genotype_pattern, action_info, re.MULTILINE)
					if genotype_match:
						genotype_from = genotype_match.group('from').strip()
						genotype_to = genotype_match.group('to').strip()
						equilibrium.genotype_from = genotype_from
						equilibrium.genotype_to = genotype_to
					else:
						print('Error! Input file indicates a mutation at equilibrium', equilibrium_key, ', but the relevant genotypes are not correctly specified!')
					frequency_pattern = '^[Ff]requency\s*=\s*(?P<frequency>[\d.e\-\+]+)$'
					frequency_match = re.search(frequency_pattern, action_info, re.MULTILINE)
					if frequency_match:
						try:
							frequency = float(frequency_match.group('frequency').strip())
							equilibrium.frequency = frequency
						except:
							print('Error! Invalid input for introduction frequency in equilibrium', equilibrium_key)
							sys.exit(1)
					else:
						print('OBS! No input for introduction frequency found for equilibrium', equilibrium_key, '. Using default frequency 0.001')
						frequency = 0.001
					deme_pattern = '^[Dd]eme\s*=\s*(?P<key>[A-Za-z\d]+)$'
					deme_match = re.search(deme_pattern, action_info, re.MULTILINE)
					if deme_match:
						deme_key = deme_match.group('key')
						try:
							deme_index = self.population.deme_keys.index(deme_key)
						except:
							print('Error!', deme_key, 'is not a valid deme key! Please revise input for equilibrium', equilibrium_key)
						equilibrium.deme_index = deme_index
					else:
						if self.population.demes_n > 1:
							print('Error! No input given for the deme in which the mutation should appear at equilibrium', equilibrium_key +'!')
							sys.exit(1)
				elif action == 'stop migration':
					print('Equilibrium migration OFF')
					equilibrium.migration = False
				
				elif action == 'start migration':
					print('Equilibrium migration ON')
					equilibrium.migration = True
				
			else:
				print('Error! No action found for equilibrium', equilibrium_key)
				sys.exit(1)
					
					
			index += 1
		
		# Read report input
		report_pattern = '^\s*#\s*[Rr]eport(?P<info>[^#]+)'
		report_match = re.search(report_pattern, input_text, re.MULTILINE)
		if report_match:
			report_info = report_match.group('info')
			alleles_pattern = '^\s*[Aa]lleles*\s*=\s*(?P<allele_keys>[A-Za-z\d,@$\s\[\]]+)$'
			haplotypes_pattern = '^\s*[Hh]aplotypes*\s*=\s*(?P<haplotype_keys>[A-Z\d,@$\s&\[\]]+)$'
			equilibria_pattern = '^\s*[Ee]quilibria\s*=\s*(?P<equilibria_tf>[A-Za-z\s]+)$'
			file_pattern = '^\s*[Ff]ile\s*=\s*(?P<file_name>[A-Za-z.\d\/\+\_\\\/\.]+)$'
			genotypes_pattern = '^\s*[Gg]enotypes\s*=\s*(?P<genotypes>[A-Za-z\d,@$\s\[\]]+)$'
			alleles_match = re.search(alleles_pattern, report_info, re.MULTILINE)
			haplotypes_match = re.search(haplotypes_pattern, report_info, re.MULTILINE)
			equilibria_match = re.search(equilibria_pattern, report_info, re.MULTILINE)
			file_match = re.search(file_pattern, report_info, re.MULTILINE)
			genotypes_match = re.search(genotypes_pattern, report_info, re.MULTILINE)
			report_equilibria = True
			
			
			if genotypes_match:
				if genotypes_match.group('genotypes').replace(' ','').lower() == 'all':
					genotype_keys = 'all'
				else:
					genotype_keys = genotypes_match.group('genotypes').replace(' ','').split(',')
			else:
				genotype_keys = None

			
			if alleles_match:
				allele_keys = alleles_match.group('allele_keys').replace(' ','').strip().split(',')
			else:
				print('OBS! No input for alleles found in report')
			
			if haplotypes_match:
				haplotype_keys = haplotypes_match.group('haplotype_keys').replace(' ','').strip().split(',')
			else:
				haplotype_keys = []
				
			
			if equilibria_match:
				equilibria_tf = equilibria_match.group('equilibria_tf').replace(' ','').lower()
				if equilibria_tf == 'true' or equilibria_tf == 't':
					report_equilibria = True
				elif equilibria_tf == 'false' or equilibria_tf == 'f':
					report_equilibria = False
			else:
				equilibria_match = None
				
			if file_match:
				file_name = file_match.group('file_name').strip()
			else:
				print('NB! No input found for output file name in section #Report')
				file_name = None
				
			report = Report(self, self.population, self.input_filename, allele_keys = allele_keys, haplotype_keys = haplotype_keys, report_equilibria = report_equilibria, file_name = file_name, input_text = input_text, genotype_keys = genotype_keys)
			self.report = report
			
		else:
			print('Error! No input given for # Report!')
			sys.exit(1)
		
		# initialize:
		self.population.initialize_diplotypes()
		self.population.initialize_genotypes()
		self.population.set_sex_keys()
		self.population.make_genotype_array()
		self.population.initialize_frequency_arrays()
		self.find_equilibria_keepers()
		self.population.initialize_couples()
		self.population.initialize_preference_tensor()
		self.population.set_genotype_frequencies(male_genotype_frequencies, female_genotype_frequencies)
		self.population.initialize_fitness_contributions()
		self.population.initialize_preference_contributions()
		self.population.calculate_fitness()
		self.population.calculate_preferences()
		self.initialize_equilibrium_follow()
		
		if self.population.load_offspring_filename is None:
			self.population.calculate_all_recombination_pattern_probabilitites()
			self.population.calculate_gamete_frequencies()
			self.population.print_gamete_frequencies()
			self.population.initialize_offspring_tensors()
			self.population.find_offspring()
		else: # load offspring tensor
			try:
				self.population.female_offspring_tensor = numpy.load('female_' + self.population.load_offspring_filename + '.npy')
				self.population.male_offspring_tensor = numpy.load('male_' + self.population.load_offspring_filename + '.npy')
			except:
				print('Error! Invalid file or file name: ' + self.population.load_offspring_filename)
				sys.exit(1)
		
		if self.population.save_offspring_filename is not None: # save offspring tensor
			numpy.save('female_' + self.population.save_offspring_filename, self.population.female_offspring_tensor)
			numpy.save('male_' + self.population.save_offspring_filename, self.population.male_offspring_tensor)

		self.initialize_equilibria()
		self.population.normalize_offspring_tensors()
		self.report.initialize()
		

	
	def add_fitness_contributions(self, fitness_info, deme):
		'''
		Adds the fitness contributions to the appropriate deme.
		
		Arguments:
		fitness_info (str): A raw string copied directly from the input file, to be processesed further in this method.
		deme: a Deme instance.
		'''
		incompatibility_pattern = '^\s*[Ii]ncompatibilities\s*=\s*(?P<info>[A-Z,.\s\d]+)$'
		incompatibility_matches = re.finditer(incompatibility_pattern, fitness_info, re.MULTILINE)
		for incompatibility_match in incompatibility_matches: # indentify input for incompatibilities
			incompatibility_info = incompatibility_match.group('info').replace(' ', '').split(',')
			deme.set_incompatibility(incompatibility_info)
		set_directly_pattern = '^\s*(?P<key>[$@A-Z\d&a-z{}\[\]]+)\s*=\s*(?P<fitness>[\d.]+)$'
		set_directly_matches = re.finditer(set_directly_pattern, fitness_info, re.MULTILINE)
		
		for set_directly_match in set_directly_matches: # identify other input
			key = set_directly_match.group('key')
			if '{m}' in key:
				key = key.replace('{m}', self.population.male_key)
			elif '{f}' in key:
				key = key.replace('{f}', self.population.female_key)
			general_key = key[::4].replace('','&')[1:-1]
			fitness = float(set_directly_match.group('fitness').strip())
			deme.add_fitness_contribution(general_key, key, fitness)
			
	def add_preference_contributions(self, mating_info, deme):
		'''
		Adds the fitness contributions to the appropriate deme.
		
		Arguments:
		mating_info (str): A raw string copied directly from the input file, to be processesed further in this method.
		deme: a Deme instance.
		'''
		
		quickset_pattern = '^\s*[Qq]uickset\s*=\s*(?P<info>[A-Z,.\s\d]+)$'
		quickset_matches = re.finditer(quickset_pattern, mating_info, re.MULTILINE)
		for quickset_match in quickset_matches:
			quickset_info = quickset_match.group('info').replace(' ','').split(',')
			deme.quickset_mating_preferences(quickset_info)
	
	def run(self):
		'''
		Runs the simulation.
		'''
		population = self.population
		next_equilibrium = self.equilibria[0]
		stop = False
		if next_equilibrium.load_frequencies_filename is not None: # load initial frequencies (optional)
			male_frequencies = numpy.load('male_' + next_equilibrium.load_frequencies_filename + '.npy')
			female_frequencies = numpy.load('female_' + next_equilibrium.load_frequencies_filename +'.npy')
		else:
			male_frequencies = population.male_frequencies
			female_frequencies = population.female_frequencies
			
		self.report.add_equilibrium_frequencies(male_frequencies, female_frequencies) # store initial frequencies
		migration_matrix = population.migration_matrix.transpose() # note that the migration matrix is transposed
		dot = numpy.dot
		array = numpy.array
		outer = numpy.multiply.outer
		demes_n = population.demes_n
		c = self.population.c # cost of searching
		check_frequency = next_equilibrium.check_frequency # the number of generations between each time the program checks if the equilibrium is reached
		equilibrium_count = 0 # counts the number of equilibria reached
		generation = 0 # counts the total number of generations
		current_equilibrium_generation = 0 # counts the number of generations since the last equilibrium
		next_equilibrium.start_generation = generation 
		following = {}
		nsum = numpy.sum
		check_next = True
		self.report.set_time(start = True)
		cp = copy.copy
		migration_on = True  # make user set initial migration switch
		array = numpy.array
		male_frequencies = array([numpy.delete(male_frequencies[d], next_equilibrium.male_remove_indices) for d in range(demes_n)]) # remove the male genotypes indicated in the input file
		female_frequencies = array([numpy.delete(female_frequencies[d], next_equilibrium.female_remove_indices) for d in range(demes_n)])
		
		old_male_frequencies = cp(male_frequencies) # used to check if static demes has reached equilibrium (see class Deme, attribute self.static)
		old_female_frequencies = cp(female_frequencies)
	
		male_fitness = next_equilibrium.male_fitness_reduced # the fitness of eah male in each deme
		female_fitness = next_equilibrium.female_fitness_reduced # the fitness of each female in each deme
		
		# genotype keys:
		male_keys = next_equilibrium.male_keys_reduced
		female_keys = next_equilibrium.female_keys_reduced
	
		potential_dynamic_deme_indices = array(population.dynamic_deme_indices) # the indices of static demes (that has not necessairly reached equilibrium yet)
		potential_static_deme_indices = array(population.static_deme_indices) # the indices of dynamic demes
		
		
		dynamic_deme_indices = array(range(demes_n))
		static_deme_indices = array([])
		
		dynamic_deme_tf = array([True for i in range(demes_n)])
		static_deme_tf = array([False for i in range(demes_n)])
		
		
		
		if list(static_deme_indices) != list(potential_static_deme_indices):
			check_static = True
		else:
			check_static = False
			
		preference_tensor = next_equilibrium.preference_tensor_reduced # the preference of each female for each male in each deme (the p_ab values from chapter 3 in the thesis)
		
		preference_tensor_dynamic = preference_tensor[dynamic_deme_indices] # the preferences in only dynamic demes
		
		male_offspring_tensor = next_equilibrium.male_offspring_tensor_reduced # the proportion of each male genotype (among all male genotypes) produced by each couple
		female_offspring_tensor = next_equilibrium.female_offspring_tensor_reduced # the proportion of each female genotype (among all female genotypes) produced by each couple
		
		male_zeros = self.population.male_zeros # just an array of zeros
		female_zeros = self.population.female_zeros
		
		male_keeper_indices = next_equilibrium.male_keeper_indices # the indices of males that are not to be removed
		female_keeper_indices = next_equilibrium.female_keeper_indices
		
		
		# some boolean variables that determine whether certain computional shortcuts are applicable:
		if male_keys == female_keys and (male_offspring_tensor == female_offspring_tensor).all():
			offspring_same = True
		else:
			offspring_same = False
			
		if offspring_same and (male_fitness == female_fitness).all():
			selection_same = True
		else:
			selection_same = False
			
		if offspring_same and (male_frequencies == female_frequencies).all():
			migration_same = True
		else:
			migration_same = False
		

		
		stop = False
		next_equilibrium.check(male_frequencies, female_frequencies, dynamic_deme_indices, generation)
		
		dynamic_demes_n = demes_n # current number of dynamic demes + static demes that havenot yet reached equilibrium
		
		initial = False
		
		
		while not stop:
			generation += 1
			current_equilibrium_generation += 1
			
			# migration:
			if migration_on: 
				male_frequencies = migration_matrix.dot(male_frequencies)
				if migration_same:
					female_frequencies = cp(male_frequencies)
				else:
					female_frequencies = migration_matrix.dot(female_frequencies)
					
			# selection:
			male_frequencies[dynamic_deme_indices] = male_frequencies[dynamic_deme_indices]*male_fitness[dynamic_deme_indices]
			
			if selection_same:
				female_frequencies = cp(male_frequencies)
			else:
				female_frequencies[dynamic_deme_indices] = female_frequencies[dynamic_deme_indices]*female_fitness[dynamic_deme_indices]
			
			# normalize
			male_frequencies = (male_frequencies.transpose()/nsum(male_frequencies, axis=1)).transpose()
			female_frequencies = (female_frequencies.transpose()/nsum(female_frequencies, axis=1)).transpose()
			
			# calculate couple frequencies:
			couple_frequencies = array([outer(male_frequencies[d], female_frequencies[d]) for d in dynamic_deme_indices])*preference_tensor_dynamic
			N = array([male_frequencies[d].dot(preference_tensor[d]) for d in dynamic_deme_indices])
			if c != 0:
				N = N + c*(1-N)
			dynamic_couple_frequencies = array([(couple_frequencies[d]/N[d]).ravel() for d in range(dynamic_demes_n)])
			
			# calculate next generation frequencies
			dynamic_male_frequencies = dynamic_couple_frequencies.dot(male_offspring_tensor)
			
			if offspring_same:
				dynamic_female_frequencies = cp(dynamic_male_frequencies)
			else:
				dynamic_female_frequencies = dynamic_couple_frequencies.dot(female_offspring_tensor)
			
			# normalize:
			if c!= 0:
				dynamic_male_frequencies = (dynamic_male_frequencies.transpose()/nsum(dynamic_male_frequencies, axis=1)).transpose()
				dynamic_female_frequencies = (dynamic_female_frequencies.transpose()/nsum(dynamic_female_frequencies, axis=1)).transpose()
			
			male_frequencies[dynamic_deme_indices] = dynamic_male_frequencies
			female_frequencies[dynamic_deme_indices] = dynamic_female_frequencies
				
			
			if current_equilibrium_generation % check_frequency == 0 or check_next: # if True, check if equilibrium is reached
			
				if next_equilibrium.print_follow:
					print('\nGeneration '+str(generation))
					print(time.strftime('%c'))

				if next_equilibrium.check(dynamic_male_frequencies, dynamic_female_frequencies, dynamic_deme_indices, generation): # True if equilibrium is reached
					male_frequencies_full = cp(male_zeros)
					female_frequencies_full = cp(female_zeros)
					
					male_frequencies_full[:,male_keeper_indices] = male_frequencies
					female_frequencies_full[:,female_keeper_indices] = female_frequencies
					
					if next_equilibrium.save_frequencies_filename is not None: # save genotype frequencies (optional)
						male_save_frequencies_filename = 'male_' + next_equilibrium.save_frequencies_filename
						female_save_frequencies_filename = 'female_' + next_equilibrium.save_frequencies_filename							
						numpy.save(male_save_frequencies_filename, male_frequencies_full)
						numpy.save(female_save_frequencies_filename, female_frequencies_full)
					
					self.report.add_equilibrium_frequencies(male_frequencies_full, female_frequencies_full)
					if self.report.report_equilibria:
						self.report.update_equilibria_text(generation, next_equilibrium, male_frequencies_full, female_frequencies_full)
					if equilibrium_count == len(self.equilibria)-1: # final equilibrium reached
						self.report.set_time(start = False)
						stop = True
						check_static = False
					else:
						if next_equilibrium.action == 'mutate': # introduce new mutation
							new_frequencies = next_equilibrium.execute_action(male_frequencies_full, female_frequencies_full)
							male_frequencies = new_frequencies[0]
							female_frequencies = new_frequencies[1]
						
						else:
							male_frequencies = male_frequencies_full
							female_frequencies = female_frequencies_full
						
						migration_on = next_equilibrium.migration # turn migration on/off
						
						if next_equilibrium.new_migration_matrix is not None: # set new migration matrix
							population.migration_matrix = next_equilibrium.new_migration_matrix
							migration_matrix = population.migration_matrix.transpose()
							
						equilibrium_count += 1
						next_equilibrium = self.equilibria[equilibrium_count] # next equilibrium
						next_equilibrium.start_generation = generation
						
						male_frequencies = array([numpy.delete(male_frequencies[d], next_equilibrium.male_remove_indices) for d in range(demes_n)])
						female_frequencies = array([numpy.delete(female_frequencies[d], next_equilibrium.female_remove_indices) for d in range(demes_n)])
		
						old_male_frequencies = cp(male_frequencies) # used to check if static demes has reached equilibrium
						old_female_frequencies = cp(female_frequencies)
	
						male_fitness = next_equilibrium.male_fitness_reduced
						female_fitness = next_equilibrium.female_fitness_reduced
		
						male_keys = next_equilibrium.male_keys_reduced
						female_keys = next_equilibrium.female_keys_reduced
						
						dynamic_deme_indices = array(range(demes_n))
						static_deme_indices = array([])
		
						dynamic_deme_tf = array([True for i in range(demes_n)])
						static_deme_tf = array([False for i in range(demes_n)])
						
						if list(static_deme_indices) != list(potential_static_deme_indices):
							check_static = True
						else:
							check_static = False
			
						preference_tensor = next_equilibrium.preference_tensor_reduced
		
						preference_tensor_dynamic = preference_tensor[dynamic_deme_indices]
		
						male_offspring_tensor = next_equilibrium.male_offspring_tensor_reduced
						female_offspring_tensor = next_equilibrium.female_offspring_tensor_reduced
						
						male_keeper_indices = next_equilibrium.male_keeper_indices
						female_keeper_indices = next_equilibrium.female_keeper_indices
		
		
						if male_keys == female_keys and (male_offspring_tensor == female_offspring_tensor).all():
							offspring_same = True
						else:
							offspring_same = False
			
						if offspring_same and (male_fitness == female_fitness).all():
							selection_same = True
						else:
							selection_same = False

		
						dynamic_demes_n = demes_n
						
						initial = True
						
						next_equilibrium.check(male_frequencies, female_frequencies, dynamic_deme_indices, generation)
						current_equilibrium_generation = 0
					check_frequency = next_equilibrium.check_frequency
				else:
					check_next = not check_next
			
			if check_static: # check if all static demes have reached equilibrium
				print('checking static...')
				
				if not initial:
					print('checking static, generation', generation)
					print('male keys', male_keys)
					print('female keys', female_keys)
					print('old male frequencies', old_male_frequencies)
					print('new male frequencies', male_frequencies)
					print('old female frequencies', old_female_frequencies)
					print('new female frequencies', female_frequencies)
					
					print('male same', abs(old_male_frequencies-male_frequencies) < 1.0e-12)
					print('male changed', numpy.array(male_keys)[(abs(old_male_frequencies-male_frequencies) > 1.0e-12)[0]])
					print('female same', abs(old_female_frequencies-female_frequencies) < 1.0e-12)
					print('female changed', numpy.array(female_keys)[(abs(old_female_frequencies-female_frequencies) > 1.0e-12)[0]])
				
					static_demes_male = (abs(old_male_frequencies-male_frequencies) < 1.0e-12).all(axis=1)
					static_demes_female = (abs(old_female_frequencies-female_frequencies) < 1.0e-12).all(axis=1)
				
				
					new_static_deme_tf = static_demes_male*static_demes_female
				
					static_deme_tf[dynamic_deme_indices] = new_static_deme_tf
				
					static_deme_indices = numpy.where(static_deme_tf)[0]
					dynamic_deme_indices = numpy.where(static_deme_tf == False)[0]
				
					dynamic_demes_n = len(dynamic_deme_indices)
					
					preference_tensor_dynamic = preference_tensor[dynamic_deme_indices]
					
					if list(static_deme_indices) != list(potential_static_deme_indices):
						check_static = True
						old_male_frequencies = cp(male_frequencies)
						old_female_frequencies = cp(female_frequencies)
					else:
						print('ALL POTENTIALLY STATIC DEMES ARE NOW STATIC')
						check_static = False
					
				else:
					initial = False
						
		self.report.finalize()
		
		

class Calculator:
	'''
	A class with miscellaneous methods used for calculation
	'''
	def __init__(self, population):
		self.population = population
		self.memory = {'f':{}}
	
	def reverse(self, keys, left_breakpoint, right_breakpoint):
		'''
		Returns loci keys with the order inside the inverted region reversd
		'''
		l = left_breakpoint
		r = right_breakpoint
		return keys[0:l+1]+keys[l+1:r][::-1]+keys[r:]
	
	def f(self, r, q, l):
		'''
		The function f as defined in theorem 2 in thesis
		'''
		if (r,q,l) in self.memory['f']:
			return self.memory['f'][(r,q,l)]
		else:
			result = 0.0
			for j in range(q):
				r = float(r)
				q = float(q)
				result += (exp( cos(2*pi*j/q)*l ) * cos( sin(2*pi*j/q)*l - (2*pi*r*j/q) ))
			result *= (1.0/q)
			self.memory['f'][(r,q,l)] = result
			return result
	
	def w(self, c, l, m):
		'''
		Not currently used
		'''
		if c<(m+1):
			return self.poisson(l, c)
		else:
			return exp(-l)*self.f(c-(m+1), m+1, l)-self.poisson(l, c-(m+1))
		
	def matrix_power(self, matrix, power, memory_slot = None):
		'''
		Returns the matrix (argument) to the given power (argument). Matrix powers can be stored in memory for faster future computation
			by passing a hashable key to the argument memory_slot.
		'''
		if not memory_slot == None:
			if power == 0:
				identity = numpy.identity(matrix.shape[0])
				if memory_slot not in memory:
					self.memory[memory_slot] = {0: identity}
				return identity
			elif power == 1:
				if memory_slot not in memory:
					self.memory[memory_slot] = {1: matrix}
				return matrix
			else:
				if memory_slot in self.memory and power in self.memory[memory_slot]:
					return self.memory[memory_slot][power]
				if memory_slot in self.memory and power-1 in self.memory[memory_slot]:
					new_matrix = numpy.dot(self.memory[memory_slot][power-1], matrix)
					self.memory[memory_slot][power] = new_matrix
					return new_matrix
				else:
					old_matrix = matrix
					for i in range(power-1):
						new_matrix = numpy.dot(old_matrix, matrix)
						old_matrix = new_matrix
					if memory_slot in self.memory:
						memory[memory_slot][power] = new_matrix
					else:
						memory[memory_slot] = {power: new_matrix}
					return new_matrix
		
		else:
	
			if matrix.shape[0]!= matrix.shape[1]:
				print('Error!')
				return 0
			old_matrix = matrix
			if power == 0:
				identity = numpy.identity(matrix.shape[0])
				return identity
		
			elif power == 1:
				return matrix
			else:
				for i in range(power-1):
					new_matrix = numpy.dot(old_matrix, matrix)
					old_matrix = new_matrix
				return new_matrix
	
	def poisson(self, l, x):
		return (exp(-l)*l**x)/factorial(x)		


class Report:
	'''
	A class for storing and manipulation information relating to a simulation report.
	
	Key attributes:
	selt.male_equilibria_frequencies/female_equilibria_frequencies: an annary with the frequencies of each male/female genotype at each equilibrium.
	self.start_time/self.end_time: the time at which the simulation started/ended.
	
	'''
	def __init__(self, simulation, population, input_filename, allele_keys = None, haplotype_keys = None, report_equilibria = True, frequency = None, file_name = None, input_text = None, genotype_keys = None):
		self.simulation = simulation
		self.population = population
		self.input_filename = input_filename
		self.allele_keys = allele_keys
		self.haplotype_keys = haplotype_keys
		self.report_equilibria = report_equilibria
		self.frequency = frequency
		self.output_filename = file_name
		self.population_info = ''
		self.equilibria_text = ''
		self.final_text = ''
		self.input_text = input_text
		self.genotype_keys = genotype_keys
		self.male_equilibria_frequencies = []
		self.female_equilibria_frequencies = []
		self.male_genotype_keys = []
		self.female_genotype_keys = []
		self.total_genotype_keys = []
		self.male_genotype_indices = []
		self.female_genotype_indices = []

	
	def add_equilibrium_frequencies(self, male_frequencies, female_frequencies):
		self.male_equilibria_frequencies.append(male_frequencies)
		self.female_equilibria_frequencies.append(female_frequencies)
	
	def set_time(self, start = True):
		current_time = time.strftime('%d') + '/' + time.strftime('%m') + '/' + time.strftime('%Y') + ' - ' + time.strftime('%H') + ':' + time.strftime('%M') + ':' + time.strftime('%S')
		if start:
			self.start_time = current_time
		else:
			self.end_time = current_time
		
	def initialize(self):
		'''
		Find the alleles, haplotypes and genotypes whose frequencies shall be included in the report.
		'''
		alleles = []
		subtypes = []
		self.input_filename = self.simulation.input_filename
		for allele_key in self.allele_keys:
			alleles.append(self.population.alleles_dictionary[allele_key])
		for haplotype_key in self.haplotype_keys:
			subtype = Subtype(self.population, haplotype_key)
			for haplotype in self.population.haplotypes:
				if set(haplotype_key.split('&')).issubset(haplotype.locus_allele_keys):
					subtype.add_indices(haplotype.male_indices, male = True)
					subtype.add_indices(haplotype.female_indices, male = False)
			subtypes.append(subtype)
		
		if self.genotype_keys == 'all':
			self.male_genotype_keys = self.population.male_keys
			self.female_genotype_keys = self.population.female_keys
			self.male_genotype_indices = range(len(self.male_genotype_keys))
			self.female_genotype_indices = range(len(self.female_genotype_keys))
		elif self.genotype_keys != None:
			for genotype_key in self.genotype_keys:
				if self.population.male_key in genotype_key:
					add_male = True
					add_female = False
					new_genotype_key = genotype_key.replace(self.population.male_key, '')
				elif self.population.female_key in genotype_key:
					add_male = False
					add_female = True
					new_genotype_key = genotype_key.replace(self.population.female_key, '')
				else:
					add_male = True
					add_female = True
					new_genotype_key = genotype_key
				error = False
				if add_male:
					if new_genotype_key in self.population.male_keys:
						self.male_genotype_keys.append(new_genotype_key)
						self.male_genotype_indices.append(self.population.male_keys.index(new_genotype_key))
					else:
						error = True
					
				if add_female:
					if new_genotype_key in self.population.female_keys:
						self.female_genotype_keys.append(new_genotype_key)
						self.female_genotype_indices.append(self.population.female_keys.index(new_genotype_key))
					else:
						error = True
				
				if error:
					print('Error! ' + genotype_key + ' is not a valid genotype key! Please revise input section #Report.')
					sys.exit(1)
						
						
		self.alleles = alleles
		self.subtypes = subtypes
	
	
	def update_equilibria_text(self, generation, equilibrium, male_frequencies, female_frequencies):
		new_equilibria_text = '\nGeneration ' + str(generation) + ': Equilibrium ' + equilibrium.key + ' reached.'
		new_equilibria_text += '\nAllele frequencies:\n'
		for allele in self.alleles:
			new_equilibria_text += allele.key + ': ' + str(allele.find_frequencies(male_frequencies, female_frequencies)) + '\n'
		new_equilibria_text += 'Subtype frequencies\n'
		for subtype in self.subtypes:
			new_equilibria_text += subtype.key + ': ' + str(subtype.find_frequencies(male_frequencies, female_frequencies)) + '\n'
		
		print(new_equilibria_text)
		self.equilibria_text += new_equilibria_text
	
	def finalize(self):
		self.final_text += '\nSIMULATION FOR INPUT FILE ' + self.input_filename
		self.final_text += '\n\nSimulation began: ' + self.start_time
		self.final_text += '\nSimulation ended: ' + self.end_time
		
		self.final_text += '\n\nRAW INPUT FILE: \n'
		self.final_text += self.input_text
		
		self.final_text += '\n\nCHROMOSOMES:\n'
		for chromosome in self.population.chromosomes:
			self.final_text += '\nChromosome ' + chromosome.key + ':'
			if chromosome.sex:
				if chromosome.heteromorphic:
					self.final_text += '\n  Heteromorphic sex chromosome'
				else:
					self.final_text += '\n  Homomorphic or imperfectly hetermorphic sex chromosome'
				if chromosome.male_heterogametic:
					self.final_text += '\n  Male heterogametic'
				else:
					self.final_text += '\n  Female heterogametic'
			self.final_text += '\n  Loci: ' + chromosome.loci_keys
			if chromosome.sex:
				if len(chromosome.microlinked_loci_keys) > 0:
					microlinked_loci = chromosome.microlinked_loci_keys
				else:
					microlinked_loci = 'None'
				if len(chromosome.macrolinked_loci_keys) > 0:
					macrolinked_loci = chromosome.macrolinked_loci_keys
				else:
					macrolinked_loci = 'None'
				self.final_text += '\n  Microlinked loci: ' + str(microlinked_loci)
				self.final_text += '\n  Macrolinked loci: ' + str(macrolinked_loci)
			if len(chromosome.lambda_values) > 0:
				self.final_text += '\n  Lambda values: ' + str(chromosome.lambda_values).replace('[', '(').replace(']', ')')
			if len(chromosome.mu_values) > 0:
				self.final_text += '\n  Mu values: ' + str(chromosome.mu_values).replace('[', '(').replace(']', ')')
					
			if chromosome.inversion:
				self.final_text += '\n  d values: ' + str(chromosome.d_values).replace('[', '(').replace(']', ')')
			
			self.final_text += '\n'
		self.final_text += '\nGamma values: ' + str(self.population.gamma_values).replace('[', '(').replace(']', ')')
		if self.population.fitness_interaction is not None:
			self.final_text += '\nFitness interaction: ' + self.population.fitness_interaction
		else:
			self.final_text += '\nFitness interaction: multiplicative (default)'
		if self.population.preference_interaction is not None:
			self.final_text += '\nPreference interaction: ' + self.population.preference_interaction
		else:
			self.final_text += '\nPreference interaction: multiplicative (default)'
		
		
		if len(self.population.demes) > 0:
			self.final_text += '\n MIGRATION :\n'
			self.final_text +=  str(self.population.migration_matrix)
			self.final_text += '\n'
				
		self.final_text += '\nEQUILIBRIA:\n'
		for equilibrium in self.simulation.equilibria:
			self.final_text += '\nEquilibrium ' + equilibrium.key
			self.final_text += '\n  Checked every ' + str(equilibrium.check_frequency) + ' generation'
			self.final_text += '\n  Equilibrium reached after ' + str(equilibrium.reached_at_generation) + ' generations'
			if equilibrium.action == 'mutate':
				self.final_text += '\n  When reached, mutate ' + equilibrium.genotype_from + ' to ' + equilibrium.genotype_to + ' with starting frequency ' + str(equilibrium.frequency)
			elif equilibrium.action == 'end':
				self.final_text += '\n  When reached, end'
		
			self.final_text += '\n'
		
		self.final_text += '\n\nDEMES:\n'
		for deme in self.population.demes:
			self.final_text += '\nDeme ' + deme.key + ':'
			if len(deme.fitness_contributions) > 0:
				self.final_text += '\n  Fitness contributions:'
				for general_key, specific_dictionary in deme.fitness_contributions.items():
					self.final_text += '\n    ' + general_key
					for specific_key, fitness in specific_dictionary.items():
						self.final_text += '\n      ' + specific_key + ': ' + str(fitness)
				self.final_text += '\n'
			if len(deme.preference_contributions) > 0:
				self.final_text += '\n  Preference contributions:'
				for general_key, specific_dictionary in deme.preference_contributions.items():
					self.final_text += '\n    ' + general_key
					for specific_key, preference in specific_dictionary.items():
						self.final_text += '\n      ' + specific_key + ': ' + str(preference)
						
			self.final_text += '\nINITIAL AND EQUILIBRIA FREQUENCIES:'
			sexlocus_diplotype_index = self.population.sexlocus_diplo_index
			male_sex_key = self.population.male_key
			female_sex_key = self.population.female_key
			if len(self.male_genotype_indices) > 0:
				self.final_text += '\nMale genotypes: '
				for m in self.male_genotype_indices:
					male_genotype_key = self.population.male_keys[m][0:sexlocus_diplotype_index]+male_sex_key+self.population.male_keys[m][sexlocus_diplotype_index:]
					self.final_text += '\n  ' + male_genotype_key + ': ' + str([self.male_equilibria_frequencies[i][deme.index][m] for i in range(len(self.male_equilibria_frequencies))])
			if len(self.female_genotype_indices) > 0:
				self.final_text += '\nFemale genotypes: '
				for f in self.female_genotype_indices:
					female_genotype_key = self.population.female_keys[f][0:sexlocus_diplotype_index]+female_sex_key+self.population.female_keys[f][sexlocus_diplotype_index:]
					self.final_text += '\n  ' + female_genotype_key + ': ' + str([self.female_equilibria_frequencies[i][deme.index][f] for i in range(len(self.female_equilibria_frequencies))])
			if len(self.alleles) > 0:
				self.final_text += '\nAlleles: '
				for allele in self.alleles:
					self.final_text += '\n  ' + allele.key + ': ' + str([allele.find_frequencies(self.male_equilibria_frequencies[i], self.female_equilibria_frequencies[i], print_to_screen = False)[deme.index] for i in range(len(self.male_equilibria_frequencies))])
			if len(self.subtypes) > 0:
				self.final_text += '\nHaplotypes: '
				for haplotype in self.subtypes:
					self.final_text += '\n  ' + haplotype.key + ': ' + str([haplotype.find_frequencies(self.male_equilibria_frequencies[i], self.female_equilibria_frequencies[i], print_to_screen = False)[deme.index] for i in range(len(self.male_equilibria_frequencies))])	
			self.final_text += '\n\n************************\n\n'
			self.final_text += '\n'
		
		if self.output_filename is not None:
			if self.output_filename[0] == '+':
				dot_index = self.input_filename.index('.')
				self.output_filename = self.input_filename[0:dot_index]+self.output_filename[1:]
				print('output_filename', self.output_filename)
			outfile = open(self.output_filename, 'a')
			outfile.write(self.final_text)
			outfile.close()
		print(self.final_text)
			
		
					

class Subtype:
	'''
	A class for storing and manipulating information relating to a single subtype (a haplotype that may not include an allele for all loci on a chromosome)
	
	Key attributes:
	self.male_indices/self.female_indices: a list of indices of males/females that has the subtype
	'''
	def __init__(self, population, key):
		self.key = key
		self.male_indices = []
		self.female_indices = []
		self.population = population
		microlinked = True
		macrolinked = True
		for allele_key in key.split('&'):
			try:
				if microlinked and not self.population.alleles_dictionary[allele_key].locus.microlinked:
					microlinked = False
				if macrolinked and not self.population.alleles_dictionary[allele_key].locus.macrolinked:
					macrolinked = False
				if not microlinked and not macrolinked:
					break
			except:
				print('Error!', key, ' is not a valid input for haplotype.')
				sys.exit(1)
		
		self.microlinked = microlinked
		self.macrolinked = macrolinked

	def add_indices(self, indices, male):
		if male:
			self.male_indices += indices
		else:
			self.female_indices += indices
	
	def find_frequencies(self, male_frequencies, female_frequencies, print_to_screen = True):
		if self.microlinked: 
			factor = 1.0/0.25
		elif self.macrolinked:
			factor = 1.0/0.75
		else:
			factor = 1.0
		f = factor*0.25*(numpy.sum(male_frequencies[:,self.male_indices], axis = 1) + numpy.sum(female_frequencies[:,self.female_indices], axis = 1))
		if print_to_screen:
			print(self.key)
			print(f)
		return f
	



	
	
		
	
	
