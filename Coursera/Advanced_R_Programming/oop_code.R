library(readr)
library(magrittr)
library(dplyr)
library(tibble)

##############################################################################

########### S4 ############

### Longitudinal Class ###

setClass("LongitudinalData", 
         slots = list(original_df = "tbl_df",
                      id = "numeric",
                      visitors = "numeric", 
                      room = "character",
                      value = "numeric",
                      timepoint = "numeric",
                      unique_patients = "numeric",
                      patient_df_list = "list"))
  
setGeneric("make_LD", function(x){
  standardGeneric("make_LD")
})

setMethod("make_LD", 
          c(x = "list"),
          function(x){
            data_df <- x
            patients <- unique(data_df$id)
            list_of_patient_dataFrames <- vector(mode = "list", length = length(patients))
            
            # Below, I am iterating through the patients by their ID,
            # then I will create separate dataframe instances for each patient and store in a list
            
            for (i in 1:length(patients)){
              temp_df <- tibble(filter(data_df, data_df$id == patients[[i]]))
              list_of_patient_dataFrames[[i]] <- temp_df
            }
            
            # Below, I am just grabbing the relevant info in local variables to create my new LD object
            
            id_nums <- data_df$id  
            num_visitors <- data_df$visit
            room_type <- data_df$room
            pollutant_values <- data_df$value
            timepoint_values <- data_df$timepoint
            
            LD <- new("LongitudinalData",
                      original_df = data_df,
                      id = id_nums,
                      visitors = num_visitors,
                      room = room_type,
                      value = pollutant_values,
                      timepoint = timepoint_values,
                      unique_patients = patients,
                      patient_df_list = list_of_patient_dataFrames
                      
            )
            return(LD)
          })

## creating a method for the parent class to take in a list, like read.csv(MIE.csv) and return the equivalent ##

setGeneric("subject", function(df, patient_id){
  standardGeneric("subject")
})

setMethod("subject",
          c(df = "list", patient_id = "numeric"),
          function(df, patient_id){
            if(patient_id %in% df$id == TRUE){
              filtered_df <- filter(df, id == patient_id)
              return(filtered_df)
            }
          })

## same thing with subject ##

setGeneric("visit", function(df, num_visit){
  standardGeneric("visit")
})

setMethod("visit", 
          c(df = "list", num_visit = "numeric"),
          function(df, num_visit){
            if(num_visit %in% df$visit == TRUE){
              filtered_df <- filter(df, visit == num_visit)
              return(filtered_df)
            }
            else{
              paste("Visit count '", id_nums, "' is not in the dataset.")
            }
          })

## and room ##

setGeneric("room", function(df, room_type){
  standardGeneric("room")
})

setMethod("room", c(df = "list", room_type = "character"),
          function(df, room_type){
            if(room_type %in% df$room == TRUE){
              filtered_df <- filter(df, room == room_type)
              return(filtered_df)
            }
            else{
              paste("Room type '", room_type, "' is not in the dataset.")
            }
          })

setGeneric("summary", function(LD){
  standardGeneric("summary")
})

setMethod("summary", c(LD = "LongitudinalData"),
          function(LD){
            odf <- LD@original_df
            summary_data <- aggregate(odf$value, by=list(id = odf$id, visit = odf$visit, room = odf$room), FUN = mean)
            
            # id_data <- aggregate(odf$value, by=list(id = odf$id), FUN = mean)
            # visit_data <- aggregate(odf$value, by=list(visit = odf$visit), FUN = mean)
            # room_data <- aggregate(odf$value, by=list(room = odf$room), FUN = mean)
            
            # return_data <- list(summary_data, id_data, visit_data, room_data)
            
            return(summary_data)
          })

# Above, I was thinking of providing summaries of the pollutant values with respect to the relevant
# column vectors, but I figured you could just do:
# out <- subject(myLD, 14) %>% summary() %>% filter(room == "bedroom")
# and that it would be easier to work with the entire dataset than having to remember the subscriptable order 
# of the summary items


# odf %>% group_by(id) %>% summarise_each(funs(n_distinct(.))) 
# ^^ finds count of distinct items

# aggregate(odf$value, by=list(id = odf$id, visit = odf$visit, odf$room), FUN = mean)
# ^^ aggregates by mean value for each variable in list)

# ddply(odf, c("id", "visit"), summarise, mean_value = mean(value))
# ^^ creates a column vector with combinations of each unique pair of the list("id", "visit")

setGeneric("print")

setMethod("print", 
          c(x = "LongitudinalData"), 
          function(x){
            paste("Longitudinal dataset with ", length(x@unique_patients), " patients.")
          })

##############################################################################

### Subject Class ###

setClass("subject", 
         contains = "LongitudinalData",
         slots = list(LD = "LongitudinalData"))

setGeneric("subject", function(LD, id_num){
  standardGeneric("subject")
})

setMethod("subject", c(LD = "LongitudinalData", id_num = "numeric"),
          function(LD, id_num){
            
            if(id_num %in% LD@unique_patients == TRUE){
              list_index <- match(id_num, LD@unique_patients)
              filtered_df <- LD@patient_df_list[[list_index]]
              new_LD <-make_LD(filtered_df)
              paste("Subject ID: ", id_num)
              subj_obj <- new("subject", 
                              new_LD)
              
              return(subj_obj)
              
            }
            else
              paste("Subject ", id_num, " doesn't exist")
              return(NULL)
          })

setGeneric("print")

setMethod("print",
          signature = c(x = "subject"),
          function(x){
            paste("Subject ID: ", x@unique_patients)
          })

##############################################################################


### Visit Class ###

setClass("visit", 
         contains = "LongitudinalData",
         slots = list(x = "LongitudinalData"))
setGeneric("visit", function(x, visit_count){
  standardGeneric("visit")
})
           
           
setMethod("visit", c(x = "LongitudinalData", visit_count = "numeric"),
          function(x, visit_count){
            data <- x@original_df
            if(visit_count %in% data$visit == TRUE){
              df <- filter(data, visit == visit_count)
              new_LD <- make_LD(df)
              visit_obj <- new("visit", 
                               new_LD)
              return(visit_obj)
              
            }
             else{
               paste("Amount of visitors '", visit_count, "' is not in the dataset.")
               return(NULL)
             }
           })

setGeneric("print")

setMethod("print", 
          signature = c(x = "visit"),
          function(x){
            paste("Visit count: ", unique(x@visitors))
          })

##############################################################################

### Room Class ###

setClass("room", 
         contains = "LongitudinalData",
         slots = list(LD = "LongitudinalData"))

setGeneric("room", function(x, room_type){
  standardGeneric("room")
})

setMethod("room", c(x = "LongitudinalData", room_type = "character"),
                    function(x, room_type){
                      data <- x@original_df
                      if(room_type %in% data$room == TRUE){
                        df <- filter(data, room == room_type)
                        new_LD <- make_LD(df)
                        room_obj <- new("room", 
                                        new_LD)
                        return(room_obj)
                      }
                      else{
                        paste("Room type '", room_type, "' is not in the dataset.")
                        return(NULL)
                      }
                    })

setGeneric("print")

setMethod("print", 
          signature = c(x="room"),
          function(x){
            paste("Room type: ", unique(x@room))
          })

