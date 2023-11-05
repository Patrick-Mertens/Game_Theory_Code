# Define the directory where the folders containing the CSV files are stored
base_dir <- "C:/Users/Shade/Desktop/Master/Project Game Theory Code/Downloaded file/Edited/SpiderProject_KatherLab/Exponential/Control_20_09_2023_attempt_7/"

# List all folders in the base directory
folders <- list.dirs(path = base_dir, full.names = TRUE, recursive = FALSE)

# Initialize an empty data frame to hold the combined data
combined_data <- data.frame()

# Loop through each folder
for (folder in folders) {
  # List all CSV files in the current folder
  csv_files <- list.files(path = folder, pattern = ".csv", full.names = TRUE)
  
  # Loop through each file to read and combine the data
  for (file in csv_files) {
    # Read the current CSV file
    current_data <- read.csv(file)
    
    # Combine the current data with the existing data
    combined_data <- rbind(combined_data, current_data)
  }
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
Size3_in_Increasing <- count_size_in_condition(combined_data, 'Size3', 'Inc')

#Size4
Size4_in_Up <- count_size_in_condition(combined_data, 'Size4', 'Up')
Size4in_Down <- count_size_in_condition(combined_data, 'Size4', 'Down')
Size4_in_Fluctuate <- count_size_in_condition(combined_data, 'Size4', 'Fluctuate')
Size4_in_Evolution <- count_size_in_condition(combined_data, 'Size4', 'Evolution')

####DATA FRAME TABLES TESTING
df_Size1 <- combined_data[combined_data$Size1 != "Placeholder" | combined_data$Fluctuate != "Placeholder" & combined_data$Size2 == "Placeholder" &
                          combined_data$Size3 == "Placeholder" & combined_data$Size4 == "Placeholder",]

# Extract rows where both size and Fluctuate are not placeholders, and where evolution is placeholder
df_Size1_and_Fluctuate <- combined_data[combined_data$Size1 != "Placeholder" & combined_data$Fluctuate != "Placeholder", ] #C, evolution and Flucate
df_Size2_and_Fluctuate <- combined_data[combined_data$Size2 != "Placeholder" & combined_data$Fluctuate != "Placeholder", ] #C, evolution and Flucate
df_Size2_and_Fluctuate2 <- df_Size2_and_Fluctuate[df_Size2_and_Fluctuate$Evolution == "Placeholder",]
df_Size3_and_Fluctuate <- combined_data[combined_data$Size3 != "Placeholder" & combined_data$Fluctuate != "Placeholder", ] #C, evolution and Flucate
df_Size3_and_Fluctuate2 <- df_Size3_and_Fluctuate[df_Size3_and_Fluctuate$Evolution == "Placeholder",]
df_Size4_and_Fluctuate <- combined_data[combined_data$Size4 != "Placeholder" & combined_data$Fluctuate != "Placeholder", ] #C, evolution and Flucate
df_Size4_and_Fluctuate2 <- df_Size4_and_Fluctuate[df_Size4_and_Fluctuate$Evolution == "Placeholder",]


##Extracting, patientID people that are increase and size1
df_Size1_and_Increase <- combined_data[combined_data$Size1 != "Placeholder" & combined_data$Inc != "Placeholder",]
#Freq table to double check
Freq_Size1_and_Increase <- data.frame(table(df_Size1_and_Increase$PatientID)) #They are unique

#Size2
df_Size2_and_Increase <- combined_data[combined_data$Size2 != "Placeholder" & combined_data$Inc != "Placeholder",]

#Size2
df_Size3_and_Increase <- combined_data[combined_data$Size3 != "Placeholder" & combined_data$Inc != "Placeholder",]
#Size4
df_Size4_and_Increase <- combined_data[combined_data$Size4 != "Placeholder" & combined_data$Inc != "Placeholder",]




##Saving cvs
# Set the file path
file_path <- "C:/Users/Shade/Desktop/Master/Project Game Theory Code/Downloaded file/Edited/PatientID/df_Size1_and_Increase.csv"

# Write the dataframe to a CSV file
write.csv(df_Size1_and_Increase, file = file_path, row.names = FALSE)
