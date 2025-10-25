'''
utils.py module test file

Run test with:
    pytest tests/test_utils.py -v
    pytest tests/test_utils.py --cov=utils --cov-report=html
'''
# import libraries
import pytest
import pandas as pd
import json
import yaml
from pathlib import Path
from src.utils import (
    load_csv_file,load_json_file,load_joblib_file,save_csv_file,save_json_file,save_yaml_file,
    read_yaml_file,save_joblib_file, ensure_directories, project_metadata, validate_df,
    get_memory_usage,get_timestamp,data_profile
)

class TestCSVOperations:
    '''Tests for CSV file operations'''
    def test_load_csv_file_success(self,tmp_path):
        '''Test loading a valid csv file'''
        # Arrange
        test_file = tmp_path / 'test_file.csv'
        test_data = 'Name,Age,Country\nValerie,21,Ghana\nDorcas,17,Germany'
        test_file.write_text(test_data)

        #Act
        df = load_csv_file(test_file)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2,3)
        assert df['Name'].tolist() == ['Valerie','Dorcas']
        assert df.columns.tolist() == ['Name','Age','Country']

    def test_file_not_found(self):
        '''Test that FileNotFoundError is raised for missing files'''
        with pytest.raises(FileNotFoundError):
            load_csv_file('non_existent_file.csv')

    def test_file_is_empty(self,tmp_path):
        '''Test loading an empty CSV file'''
        test_file = tmp_path / 'empty_file.csv'
        test_file.write_text("")

        with pytest.raises(pd.errors.EmptyDataError):
            load_csv_file(test_file)
    
    def save_csv_file_success(self,tmp_path):
        '''Test saving a CSV file successfully'''
        # Arrange
        df = pd.DataFrame({
            'Name' : 'Alice',
            'Age' : 14,
            'City' : 'Accra'
        })
        output_path = tmp_path /'csv_output.csv'
        # Act
        save_csv_file(df,output_path)

        assert output_path.exists()
        loaded_df = load_csv_file(output_path)
        assert df.columns.tolist() == ['Name','Age','City']
        assert df.shape == (1,3)

class TestJSONOperations:
    '''Test All JSON Operations'''
    def test_json_save_load_success(self,tmp_path):
        '''Test loading a JSON file successfully'''
        test_file = tmp_path / 'valid_file.json'
        json_data = {
            'Name' : 'Maxwell',
            'Age' : 25,
            'City' : 'Toronto',
            'Is_rich' : True
        }

        save_json_file(json_data, test_file,indent=4)
        loaded_json_data = load_json_file(test_file)
        assert loaded_json_data == json_data
        assert len(list(json_data.keys())) == 4

    def test_file_not_found_json(self):
        '''Test that a FileNotFoundError is raised if file doesn't exist'''
        with pytest.raises(FileNotFoundError):
            load_json_file('non_existent_file.json')
if __name__ == '__main__':
    pytest([__file__,'-v','--cov=utils','--cov-report=html'])