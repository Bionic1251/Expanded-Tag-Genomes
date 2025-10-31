# Define a personal library path
lib_path <- "R_library"

# Create the directory if it doesn't exist
if (!dir.exists(lib_path)) {
  dir.create(lib_path, recursive = TRUE)
}

.libPaths(lib_path)

install.packages("https://cran.r-project.org/src/contrib/Archive/nloptr/nloptr_1.2.1.tar.gz", lib = lib_path, repos = NULL, type = "source")
install.packages("lme4", repos = "https://cloud.r-project.org", lib = lib_path)

cat("All required packages installed in personal library:", lib_path, "\n")
