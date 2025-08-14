import os
import subprocess
import boto3
from boto3.s3.transfer import TransferConfig
import ray
import re
import time


@ray.remote(num_cpus=1)
class S3UploaderActor:
    """Ray actor dedicated to S3 uploads to prevent memory issues in worker processes."""
    
    def __init__(self):
        self.s3 = boto3.client('s3')

    def initialize_s3_client(self):
        self.s3 = boto3.client('s3')
    
    def upload_file(self, local_path, s3_path, config=None, delete_after_upload=False, force_gc=False, verbose=False):
        """Upload a single file to S3 using multipart upload.
        
        Args:
            local_path: Local path to file
            s3_path: S3 path (s3://bucket/key)
            config: TransferConfig object
            delete_after_upload: Whether to delete the file after successful upload
            force_gc: Force garbage collection after upload (for very large files)
            
        Returns:
            bool: True if successful, False otherwise
        """

        self.initialize_s3_client()

        if config is None:
            config = TransferConfig(
                multipart_threshold=100 * 1024 * 1024,  # 100MB threshold for multipart
                multipart_chunksize=20 * 1024 * 1024,  # 20MB chunks (reduced from 25MB)
                max_concurrency=2,  # Reduced from 4
                use_threads=True
            )
        
        # Parse S3 path to get bucket and key
        bucket, key = self._parse_s3_path(s3_path)
        if verbose:
            print(f"Uploading {local_path} to s3://{bucket}/{key}")
        
        success = False
        try:
            self.s3.upload_file(local_path, bucket, key, Config=config)
            success = True
        except Exception as e:
            print(f"Error uploading to S3: {e}")
            # Try with ml-worker profile
            try:
                boto3_session = boto3.Session(profile_name='ml-worker')
                s3_client = boto3_session.client('s3')
                s3_client.upload_file(local_path, bucket, key, Config=config)
                s3_client.close()  # Explicitly close the client
                success = True
            except Exception as e2:
                print(f"Error uploading to S3 with ml-worker profile: {e2}")
                
        # Clean up after upload
        if success and delete_after_upload:
            try:
                if os.path.exists(local_path):
                    file_size = os.path.getsize(local_path)
                    os.remove(local_path)
                    if verbose:
                        print(f"Deleted file {local_path} ({file_size/1024/1024/1024:.2f} GB)")
            except Exception as e:
                print(f"Error deleting file after upload: {e}")
                
        # Force garbage collection for very large files
        if force_gc:
            import gc
            gc.collect()
                
        return success
    
    def copy_directory(self, src_path, dst_path, exclude=None, delete_after_upload=False, verbose=False):
        """Copy a directory recursively to S3.
        
        Args:
            src_path: Local directory path to copy from
            dst_path: S3 path to copy to (s3://bucket/prefix)
            exclude: Optional regex pattern of files to exclude
            delete_after_upload: Whether to delete files after successful upload
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.initialize_s3_client()
        bucket, prefix = self._parse_s3_path(dst_path)
        prefix = prefix.rstrip('/') + '/'
        
        # Compile exclude pattern if provided
        exclude_pattern = re.compile(exclude) if exclude else None
        
        total_files = 0
        success_count = 0
        
        # Create config once outside the loop with more conservative settings
        config = TransferConfig(
            multipart_threshold=100 * 1024 * 1024,  # Increased from 50MB to 100MB
            multipart_chunksize=20 * 1024 * 1024,   # Reduced from 25MB to 20MB
            max_concurrency=1,                     # Keep at 1 for minimal resource usage
            use_threads=True                       # Changed to True for better isolation
        )
        
        # Create fallback session and client only if needed
        fallback_session = None
        fallback_client = None
        
        try:
            # Walk through the directory
            for root, _, files in os.walk(src_path):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    
                    # Skip files matching exclude pattern
                    if exclude_pattern and exclude_pattern.search(local_file_path):
                        continue
                    
                    # Calculate relative path and create corresponding S3 key
                    rel_path = os.path.relpath(local_file_path, src_path)
                    s3_key = os.path.join(prefix, rel_path).replace('\\', '/')
                    
                    total_files += 1
                    
                    try:
                        file_size = os.path.getsize(local_file_path)
                        file_size_gb = file_size/1024/1024/1024
                        if verbose:
                            print(f"Uploading {local_file_path} ({file_size_gb:.2f} GB) to s3://{bucket}/{s3_key}")
                        
                        # Add a small sleep before each upload to reduce resource contention
                        time.sleep(0.1)
                        
                        # Determine if this is a large file that needs special handling
                        force_gc = file_size > 5 * 1024 * 1024 * 1024  # 5GB threshold for GC
                        
                        self.s3.upload_file(local_file_path, bucket, s3_key, Config=config)
                        success_count += 1
                        
                        # Clean up large files after upload
                        if delete_after_upload:
                            try:
                                os.remove(local_file_path)
                                if verbose:
                                    print(f"Deleted file {local_file_path} ({file_size_gb:.2f} GB)")
                            except Exception as e:
                                print(f"Error deleting file after upload: {e}")
                                
                        # Force garbage collection for very large files
                        if force_gc:
                            import gc
                            gc.collect()
                            
                    except Exception as e:
                        print(f"Error uploading to S3: {e}")
                        # Try with ml-worker profile - create session/client only once
                        try:
                            if fallback_session is None:
                                fallback_session = boto3.Session(profile_name='ml-worker')
                                fallback_client = fallback_session.client('s3')
                            
                            fallback_client.upload_file(local_file_path, bucket, s3_key, Config=config)
                            success_count += 1
                            
                            # Clean up large files after upload
                            if delete_after_upload:
                                try:
                                    os.remove(local_file_path)
                                    if verbose:
                                        print(f"Deleted file {local_file_path} ({file_size_gb:.2f} GB)")
                                except Exception as e:
                                    print(f"Error deleting file after upload: {e}")
                                    
                            # Force garbage collection for very large files
                            if force_gc:
                                import gc
                                gc.collect()
                                
                        except Exception as e2:
                            print(f"Error uploading to S3 with ml-worker profile: {e2}")
        finally:
            # Clean up resources
            if fallback_client:
                fallback_client.close()
            if fallback_session:
                # Explicitly help garbage collection
                fallback_session = None
            
            # Final garbage collection
            import gc
            gc.collect()
            
            if verbose:
                print(f"Copied {success_count}/{total_files} files from {src_path} to {dst_path}")
        
        return success_count == total_files
    
    def _parse_s3_path(self, s3_path):
        """Parse s3://bucket/key format into bucket and key."""
        match = re.match(r's3://([^/]+)/(.+)', s3_path)
        if match:
            return match.groups()
        else:
            raise ValueError(f"Invalid S3 path: {s3_path}")
    
    def check_file_exists(self, s3_path):
        """Check if a file exists in S3."""
        bucket, key = self._parse_s3_path(s3_path)
        try:
            self.s3.head_object(Bucket=bucket, Key=key)
            return True
        except:
            try:
                # Try with ml-worker profile
                boto3_session = boto3.Session(profile_name='ml-worker')
                s3_client = boto3_session.client('s3')
                s3_client.head_object(Bucket=bucket, Key=key)
                return True
            except:
                return False


def distributed_aws_copy(src, dst, rank, recursive=False, use_s5cmd=True, exclude=None, use_ray_actor=False, max_parallel_uploads=None, delete_after_upload=False, num_processes=8, blocking=True):
    """Copy files to/from S3 using either command line tools or Ray actor.
    
    Args:
        src: Source path (local or S3)
        dst: Destination path (local or S3)
        rank: Process rank (for distributed training), only the files at index % num_processes will be uploaded from this process
        recursive: Whether to copy recursively
        use_s5cmd: Whether to use s5cmd instead of aws cli
        exclude: Pattern of files to exclude
        use_ray_actor: Whether to use Ray actors for upload
        max_parallel_uploads: Maximum number of parallel uploads
        delete_after_upload: Whether to delete files after successful upload
        num_processes: Number of processes to use for uploads
        blocking: Whether to wait for uploads to complete before returning
        
    Returns:
        If blocking=True: Boolean indicating success
        If blocking=False: Ray futures that can be waited on (for upload case) or None (for command-line case)
    """

    if dst is None:
        print(f"dst is None, skipping copy")
        return True

    if use_ray_actor:
        # Use Ray actor for S3 uploads
        # Check if src is local and dst is S3 (upload case)
        if dst.startswith("s3://"):
            if recursive:
                # Limit parallel uploads to prevent resource contention
                if max_parallel_uploads is None:
                    # Default to a conservative value
                    max_parallel_uploads = min(4, ray.available_resources().get("CPU", 8) // 4)
                
                # Create a limited pool of uploader actors
                uploaders = [S3UploaderActor.remote() for _ in range(max_parallel_uploads)]
                
                if os.path.isdir(src):
                    # Handle directory upload through a single actor for large directories
                    if os.path.getsize(src) > 10 * 1024 * 1024 * 1024:  # 10GB threshold
                        print(f"Large directory detected ({src}). Using optimized copy method.")
                        uploader = S3UploaderActor.remote()
                        future = uploader.copy_directory.remote(src, dst, exclude, delete_after_upload)
                        return ray.get(future) if blocking else future
                    
                    # Process files in batches to avoid memory issues
                    exclude_pattern = re.compile(exclude) if exclude else None
                    all_success = True
                    all_futures = []
                    
                    # Collect all files to upload
                    all_files = []
                    for root, _, files in os.walk(src):
                        for file in files:
                            local_path = os.path.join(root, file)
                            
                            # Skip files matching exclude pattern
                            if exclude_pattern and exclude_pattern.search(local_path):
                                continue
                                
                            rel_path = os.path.relpath(local_path, src)
                            s3_path = f"{dst.rstrip('/')}/{rel_path}"
                            file_size = os.path.getsize(local_path)
                            force_gc = file_size > 5 * 1024 * 1024 * 1024  # 5GB threshold for GC
                            
                            all_files.append((local_path, s3_path, file_size, force_gc))
                    
                    # Sort files by size in descending order to process larger files first
                    all_files.sort(key=lambda x: x[2], reverse=True)
                    
                    # Distribute files across processes
                    files_by_process = [[] for _ in range(num_processes)]
                    for i, file_info in enumerate(all_files):
                        process_idx = i % num_processes
                        files_by_process[process_idx].append(file_info)
                    
                    # Process files in max_batch_size chunks per process
                    max_batch_size = 20  # Process files in smaller batches
                    in_progress = []  # Track uploads in progress
                    
                    # Launch uploads for each process's files
                    for process_idx, file_group in enumerate(files_by_process):
                        # Skip empty groups
                        if not file_group or (rank % num_processes != process_idx):
                            continue
                            
                        # Get the assigned uploader for this process
                        uploader_idx = process_idx % len(uploaders)
                        uploader = uploaders[uploader_idx]
                        
                        # Process files in batches
                        for i in range(0, len(file_group), max_batch_size):
                            batch = file_group[i:i+max_batch_size]
                            
                            # Launch uploads for this batch
                            for local_path, s3_path, _, force_gc in batch:
                                future = uploader.upload_file.remote(
                                    local_path, s3_path, delete_after_upload=delete_after_upload, force_gc=force_gc
                                )
                                in_progress.append((future, local_path))
                                all_futures.append(future)
                            
                            if blocking:
                                # Wait for this batch to complete before proceeding
                                for ref, local_path in in_progress:
                                    success = ray.get(ref)
                                    if not success:
                                        all_success = False
                                
                                # Clear in_progress and force garbage collection
                                in_progress = []
                                import gc
                                gc.collect()
                    
                    return all_success if blocking else all_futures
                else:
                    # Handle single file upload - for large files this is important
                    uploader = S3UploaderActor.remote()
                    file_size = os.path.getsize(src)
                    force_gc = file_size > 5 * 1024 * 1024 * 1024  # 5GB threshold
                    future = uploader.upload_file.remote(
                        src, dst, delete_after_upload=delete_after_upload, force_gc=force_gc
                    )
                    return ray.get(future) if blocking else future
    elif use_s5cmd:
        command = ["s5cmd", "sync"]
        if recursive:
            src = src.rstrip("/") + "/*"
            dst = dst.rstrip("/") + "/"
        command += [src, dst]

        print(f"s5cmd_copy: {command}")
        try:
            subprocess.run(command, check=True)
        except:
            print(f"Error in distributed_aws_copy with command: {command}")
    else:
        command = ["aws", "s3", "cp"]
        if recursive:
            command += ["--recursive"]
        if exclude:
            command += ['--exclude', exclude]
        command += [src, dst]

        print(f"aws_copy: {command}")
        try:
            subprocess.run(command, check=True)
        except:
            print(f"Error in distributed_aws_copy with command: {command}")


def aws_copy(src, dst, recursive=False, use_s5cmd=True, exclude=None):

    if dst is None:
        print(f"dst is None, skipping aws_copy")
        return True

    if use_s5cmd:
        if recursive:
            command = ["s5cmd", "sync"]
        else:
            command = ["s5cmd", "cp"]
        if exclude:
            command += ['--exclude', exclude]
        if recursive:
            src = src.rstrip("/") + "/*"
            dst = dst.rstrip("/") + "/"
        command += [src, dst]
        
    else:
        command = ["aws", "s3", "sync"]
        if recursive:
            command += ["--recursive"]
        if exclude:
            command += ['--exclude', exclude]
        command += [src, dst]

    print(f"aws_copy: {command}")
    try:
        subprocess.run(command, check=True)
    except:
        print(f"Error in aws_copy with command: {command}")

def aws_check_file_exists(s3_path, use_ray_actor=False):
    """Check if a file exists in S3."""
    if use_ray_actor:
        uploader = S3UploaderActor.remote()
        return ray.get(uploader.check_file_exists.remote(s3_path))
    
    # Original implementation
    command = ["aws", "s3", "ls", s3_path]
    try:
        subprocess.run(command, check=True)
    except:
        return False
    return True