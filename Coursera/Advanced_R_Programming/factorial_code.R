library(purrr)
library(microbenchmark)
library(ggplot2)

##### Factorial Functions #####

# First function just a generic error checking function so I don't have to rewrite each time

check_errors<-function(x){
  if(x<0){
    paste("Error: bad factorial input!")
    stop("n should be >=0")
  }
}

# Elementary looping function for factorial, starting x and continuing until 2
# I figured eliminating one multiplication should improve performance

Factorial_loop<-function(x){
  check_errors(x)
  answer<-1
  for (i in x:2){
    answer<- answer*i
  }
  paste("Answer is:", answer)
  return(answer)
}

#Function to pass to reduce

multiply<-function(x,y){
  return(x*y)
}

# Factorial using purr's reduce

Factorial_reduce<-function(x){
  check_errors(x)
  answer <- reduce(x:1, multiply)
  return(answer)
}

# Classic recursion

Factorial_func<-function(x){
  check_errors(x)
  answer<- 1
  if(x == 0 | 1){
    answer<- 1
  }
  if(x > 1){
    answer <- x * Factorial_func(x - 1)
  }
  paste("Answer is:", answer)
  return(answer)
}

# Factorial memoization next, just did the first seven factorials and then left the rest blank

factorial_tbl <- c(1, 2, 6, 24, 120, 720, 5040, rep(NA, 43))

Factorial_mem <- function(x){
  answer <- 1
  check_errors(x)
  stopifnot(x > 0)
  if(!is.na(factorial_tbl[x])){
    answer <- factorial_tbl[x]
  }
  else {
    factorial_tbl[x-1] <<- (x-1) * Factorial_mem(x-2)
    factorial_tbl[x] <<- x * factorial_tbl[x-1]
    answer <- factorial_tbl[x]
  }
  paste("Answer is:", answer)
  return(answer)
}

# Here I wanted to make it easier to generate the figures so I wrote a function to take a numeric variable,
# and basically run a loop from 1 to the number, calculating the microbenchmark data for each iteration,
# and then store each individual time in a list so we can check performance across a range of complexities easily.

Generating_output <- function(vector, print_summaries = FALSE){
  memo_results <- vector(mode = "list", length = vector)
  loop_results <- vector(mode = "list", length = vector)
  func_results <- vector(mode = "list", length = vector)
  reduce_results <- vector(mode = "list", length = vector)
  mb_results <- vector(mode = "list", length = vector)
  
  for (i in 1:vector){
    res <- microbenchmark(
      "Recursion" = Factorial_func(i),
      "Loop" = Factorial_loop(i), 
      "Memoization" = Factorial_mem(i),
      "Reduce" = Factorial_reduce(i),
      times = 100L)
    
    memo_boolean <- res$expr[1:400] == "Memoization"
    loop_boolean <- res$expr[1:400] == "Loop"
    func_boolean <- res$expr[1:400] == "Recursion"
    reduce_boolean <- res$expr[1:400] == "Reduce"
    
    memo_times <- res$time[memo_boolean]/1000
    loop_times <- res$time[loop_boolean]/1000
    func_times <- res$time[func_boolean]/1000
    reduce_times <- res$time[reduce_boolean]/1000
    
    if(print_summaries == TRUE){
      cat("\nMemoization", i, "Summary:\n")
      print(summary(memo_times))
      cat("\nLoop" , i, "Summary:\n")
      print(summary(loop_times))
      cat("\nRecursion", i, "Summary:\n")
      print(summary(func_times))
      cat("\nReduce", i, "Summary:\n")
      print(summary(reduce_times))
    }
    
    memo_results[[i]] <- memo_times
    loop_results[[i]] <- loop_times
    func_results[[i]] <- func_times
    reduce_results[[i]] <- reduce_times
    mb_results[[i]] <- res
    
  }
  results <- c("Memoization" = memo_results, "Loop" =  loop_results, 
               "Recursion" = func_results, "Reduce" = reduce_results, "Microbenchmark Results" = mb_results)
  
  return(results)
}

Generate_autoplot <- function(number, print_output = FALSE){
  data <- Generating_output(number, print_summaries = FALSE)
  name <- paste(c("data$`Microbenchmark Results", number, "`"), collapse = "")
  mb_summary <- (eval(parse(text = name)))
  graph <- autoplot(mb_summary)
  return(data)
 
}
