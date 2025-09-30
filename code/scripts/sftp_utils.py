import pandas as pd
import paramiko
import io

def upload_df_as_csv(df, username, password, remote_path, hostname = 'mscffiles.stat.cmu.edu', port=22):
    """
    Uploads a Pandas DataFrame to a server as a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to upload.
        hostname (str): The server's IP address or hostname.
        username (str): The SSH username.
        password (str): The SSH password.
        remote_path (str): The full path for the destination file on the server
            One of the following paths:
                - /home/ktallam/News
                - /home/ktallam/Returns
        port (int, optional): The SSH port. Defaults to 22.

    Returns:
        bool: True if upload was successful, False otherwise.
    """
    ssh_client = None
    try:
        # 1. Convert DataFrame to a CSV string in memory
        print("Converting DataFrame to CSV in memory...")
        # Create an in-memory text buffer
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        # Go to the start of the buffer to read its content
        csv_buffer.seek(0)

        # 2. Establish SSH connection
        print(f"Connecting to {hostname}...")
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname, port=port, username=username, password=password)
        print("Connection successful!")

        # 3. Open SFTP session and upload the file
        sftp_client = ssh_client.open_sftp()
        print(f"Uploading to {remote_path}...")
        
        # Write the content of the in-memory buffer to the remote file
        with sftp_client.file(remote_path, 'w') as remote_file:
            remote_file.write(csv_buffer.getvalue())

        sftp_client.close()
        print("Upload complete!")
        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    finally:
        if ssh_client:
            ssh_client.close()
            print("Connection closed.")

def download_csv_as_df(username, password, remote_path, hostname = 'mscffiles.stat.cmu.edu', port=22):
    """
    Downloads a CSV file from a server and loads it into a Pandas DataFrame.

    Args:
        hostname (str): The server's IP address or hostname.
        username (str): The SSH username.
        password (str): The SSH password.
        remote_path (str): The full path to the source CSV file on the server.
            One of the following paths:
                - /home/ktallam/News
                - /home/ktallam/Returns
        port (int, optional): The SSH port. Defaults to 22.

    Returns:
        pd.DataFrame: A DataFrame containing the CSV data, or None if it fails.
    """
    ssh_client = None
    try:
        # 1. Establish SSH connection
        print(f"Connecting to {hostname}...")
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname, port=port, username=username, password=password)
        print("Connection successful!")

        # 2. Open SFTP session and read the remote file
        sftp_client = ssh_client.open_sftp()
        print(f"Downloading and reading {remote_path}...")
        
        with sftp_client.open(remote_path, 'r') as remote_file:
            # Read the file content into an in-memory text buffer
            csv_buffer = io.StringIO(remote_file.read().decode('utf-8'))
        
        sftp_client.close()

        # 3. Load the buffer into a Pandas DataFrame
        # Rewind the buffer to the beginning before reading
        csv_buffer.seek(0)
        df = pd.read_csv(csv_buffer)
        print("DataFrame created successfully!")
        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    finally:
        if ssh_client:
            ssh_client.close()
            print("Connection closed.")