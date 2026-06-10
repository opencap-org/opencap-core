def update_progress_excel(subject_num, column_name, value):
    """
    Opens an Excel file, finds the correct cell, updates it, and saves the file.
    Finds the subject by Player ID in the first column, and updates the specified column.
    """
    try:
        # Load the workbook and select the active sheet
        workbook = openpyxl.load_workbook(PROGRESS_EXCEL_FILE)
        sheet = workbook.active

        # Find the target row by looking for the subject_num in the first column (Player ID)
        target_row = None
        for row_index in range(1, sheet.max_row + 1):
            cell_value = sheet.cell(row=row_index, column=1).value
            # Handle both string and numeric comparisons
            if cell_value == subject_num or (isinstance(cell_value, (int, float)) and 
                                             isinstance(subject_num, (int, float)) and 
                                             int(cell_value) == int(subject_num)):
                target_row = row_index
                break
        
        if not target_row:
            print(f"  - EXCEL_UPDATE_WARNING: Subject ID '{subject_num}' not found in the first column of the Excel file.")
            return False

        # Find the target column by looking for the column_name in the first row
        target_col = None
        for col_index in range(1, sheet.max_column + 1):
            cell_value = sheet.cell(row=1, column=col_index).value
            if cell_value and str(cell_value).lower() == str(column_name).lower():
                target_col = col_index
                break

        if not target_col:
            print(f"  - EXCEL_UPDATE_WARNING: Column '{column_name}' not found in the header row of the Excel file.")
            return False

        # Update the cell and save the workbook
        sheet.cell(row=target_row, column=target_col).value = value
        workbook.save(PROGRESS_EXCEL_FILE)
        print(f"  - EXCEL_UPDATE: Marked '{column_name}' as '{value}' for subject {subject_num}.")
        return True

    except FileNotFoundError:
        print(f"  - EXCEL_UPDATE_ERROR: The progress file was not found at '{PROGRESS_EXCEL_FILE}'.")
        return False
    except Exception as e:
        print(f"  - EXCEL_UPDATE_ERROR: An error occurred while updating the Excel file: {e}")
        return False