# Read data from the edge list file
PARM <- read.csv("C:/Users/Shade/Downloads/parameters_time(1).csv", header = FALSE)
#The above line is from the older parms that was given

# Define the directory where the CSV files are stored
csv_dir <- "C:/Users/Shade/Desktop/Master/Project Game Theory Code/Downloaded file/Edited/SpiderProject_KatherLab/Exponential/Control_20_09_2023_attempt_7/1"

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

#Trusty way to check value of function
Size1_df <- combined_data[combined_data$Size1 != "Placeholder",] #Create a dataframe with only size1
test2 <- data.frame(table(Size1_df$PatientID))#Freq table, sum is 126



# Apply the function for each column
Size1_amount <- count_unique_patients(combined_data, 'Size1')
Size2_amount <- count_unique_patients(combined_data, 'Size2') 
Size3_amount <- count_unique_patients(combined_data, 'Size3') 
Size4_amount <- count_unique_patients(combined_data, 'Size4') 


#Trusty way to check value of function
Size2_df <- combined_data[combined_data$Size2 != "Placeholder",] #Create a dataframe with only size1
test3 <- data.frame(table(Size2_df$PatientID))

#Trusty way to check value of function
Size3_df <- combined_data[combined_data$Size3 != "Placeholder",] #Create a dataframe with only size1
test4 <- data.frame(table(Size3_df$PatientID))


#Check how many unique patietns IDs
length(data.frame(table(combined_data$PatientID))$Var1)


















##SCRAP
Size1_amount <- 0
Size2_amount <- 0

for i:length(combined_data$PatientID){
  for j:length(ncol(combined_data)){
    if(i == 1){
      Previous_ID <- combined_data$PatientID[i]
      if(combined_data[i,j] != "Placeholder"){
        #Size1 statements
        if(colnames(combined_data[i,j]) == 'Size1' &  Size1_amount == 0){
          Size1_amount <- Size1_amount + 1
          
        } else if(colnames(combined_data[i,j]) == 'Size1' &  Size1_amount > 0 & Previous_ID != combined_data$PatientID[i]){
          Size1_amount <- Size1_amount + 1
        } 
        
        #Size2 statemetns
        if(colnames(combined_data[i,j]) == 'Size2' &  Size2_amount == 0){
          Size2_amount <- Size2_amount + 1
        } else if(colnames(combined_data[i,j]) == 'Size2' &  Size2_amount == 0 & Previous_ID != combined_data$PatientID[i]){
          Size2_amount <- Size2_amount + 1
        }
          
      }
    }else if(i > 1){
      Previous_ID <- combined_data$PatientID[i-1]
      if(combined_data[i,j] != "Placeholder"){
        #Size1 statements
        if(colnames(combined_data[i,j]) == 'Size1' &  Size1_amount == 0){
          Size1_amount <- Size1_amount + 1
          
        } else if(colnames(combined_data[i,j]) == 'Size1' &  Size1_amount > 0 & Previous_ID != combined_data$PatientID[i]){
          Size1_amount <- Size1_amount + 1
        } 
        
        #Size2 statemetns
        if(colnames(combined_data[i,j]) == 'Size2' &  Size2_amount == 0){
          Size2_amount <- Size2_amount + 1
        } else if(colnames(combined_data[i,j]) == 'Size2' &  Size2_amount == 0 & Previous_ID != combined_data$PatientID[i]){
          Size2_amount <- Size2_amount + 1
        }
    }
  }
}

