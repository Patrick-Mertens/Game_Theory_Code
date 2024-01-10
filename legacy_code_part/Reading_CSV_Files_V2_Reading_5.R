# Define the directory where the CSV files are stored
csv_dir <- "C:/Users/Shade/Desktop/Master/Project Game Theory Code/Downloaded file/Edited/SpiderProject_KatherLab/Exponential/Control_attempt_7_parameters_study_5"

# List all CSV files in the directory
csv_files <- list.files(path = csv_dir, pattern = ".csv", full.names = TRUE)

# Initialize an empty data frame to hold the combined data
combined_data <- data.frame()

# Loop through each file to read and combine the data
for (file in csv_files) {
  # Read the current CSV file
  current_data <- read.csv(file)
  
  # Combine the current data with the existing data
  combined_data <- rbind(combined_data, current_data)
}



count_unique_patients <- function(df, col_name){
  unique_patients <- unique(df$PatientID[df[[col_name]] != "Placeholder"])
  length(unique_patients)
}


# Apply the function for each column
Size1_amount <- count_unique_patients(combined_data, 'Size1')
Size2_amount <- count_unique_patients(combined_data, 'Size2') 
Size3_amount <- count_unique_patients(combined_data, 'Size3') 
Size4_amount <- count_unique_patients(combined_data, 'Size4') 
Up_amount <- count_unique_patients(combined_data, 'Up')
Down_amount <- count_unique_patients(combined_data, 'Down')
Fluctuate_amount <- count_unique_patients(combined_data, 'Fluctuate')
Evolution_amount <- count_unique_patients(combined_data, 'Evolution')

# Count how many Size1 (and so on) are in Up, Down, etc.
count_size_in_condition <- function(df, size_col, condition_col){
  sum(df[[size_col]] != "Placeholder" & df[[condition_col]] != "Placeholder")
}

#Size1
Size1_in_Up <- count_size_in_condition(combined_data, 'Size1', 'Up')
Size1_in_Down <- count_size_in_condition(combined_data, 'Size1', 'Down')
Size1_in_Fluctuate <- count_size_in_condition(combined_data, 'Size1', 'Fluctuate')
Size1_in_Evolution <- count_size_in_condition(combined_data, 'Size1', 'Evolution')

#Size2
Size2_in_Up <- count_size_in_condition(combined_data, 'Size2', 'Up')
Size2_in_Down <- count_size_in_condition(combined_data, 'Size2', 'Down')
Size2_in_Fluctuate <- count_size_in_condition(combined_data, 'Size2', 'Fluctuate')
Size2_in_Evolution <- count_size_in_condition(combined_data, 'Size2', 'Evolution')

#Size3
Size3_in_Up <- count_size_in_condition(combined_data, 'Size3', 'Up')
Size3_in_Down <- count_size_in_condition(combined_data, 'Size3', 'Down')
Size3_in_Fluctuate <- count_size_in_condition(combined_data, 'Size3', 'Fluctuate')
Size3_in_Evolution <- count_size_in_condition(combined_data, 'Size3', 'Evolution')


#Size4
Size4_in_Up <- count_size_in_condition(combined_data, 'Size4', 'Up')
Size4in_Down <- count_size_in_condition(combined_data, 'Size4', 'Down')
Size4_in_Fluctuate <- count_size_in_condition(combined_data, 'Size4', 'Fluctuate')
Size4_in_Evolution <- count_size_in_condition(combined_data, 'Size4', 'Evolution')
