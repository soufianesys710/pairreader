from chainlit.data.base import BaseDataLayer
from chainlit.context import context
from chainlit.element import Element, ElementDict
from chainlit.step import StepDict
from chainlit.user import PersistedUser, User
from chainlit.types import (
    Feedback,
    PaginatedResponse,
    Pagination,
    ThreadDict,
    ThreadFilter,
    PageInfo,
)
from dataclasses import Field
from typing import Dict, List, Optional
import logging
import uuid
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("InMemoryDataLayer")

class InMemoryDataLayer(BaseDataLayer):
    """Persistence in just python objects and lists"""

    def __init__(self, verbosity: bool = False):
        self.verbosity = verbosity
        self.users: List["PersistedUser"] = []
        self.elements: List["ElementDict"] = []
        self.steps: List["StepDict"] = []
        self.threads: List["ThreadDict"] = []
        self.feedbacks: List[Feedback] = []
        if self.verbosity:
            logger.info("InMemoryDataLayer initialized")

    async def get_user(self, identifier: str) -> Optional["PersistedUser"]:
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - get_user called with identifier: {identifier}")
        user = next((u for u in self.users if u.identifier == identifier), None)
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - get_user returning: {user}")
        return user

    async def create_user(self, user: "User") -> Optional["PersistedUser"]:
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - create_user called with user: {user}")
        # Create PersistedUser from User with required fields
        persisted_user = PersistedUser(
            id=str(uuid.uuid4()),
            createdAt=datetime.now().isoformat(),
            identifier=user.identifier,
            metadata=user.metadata
        )
        self.users.append(persisted_user)
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - create_user created user with id: {persisted_user.id}")
        return persisted_user

    async def delete_feedback(self, feedback_id: str) -> bool:
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - delete_feedback called with feedback_id: {feedback_id}")
        initial_count = len(self.feedbacks)
        self.feedbacks = [f for f in self.feedbacks if getattr(f, 'id', None) != feedback_id]
        final_count = len(self.feedbacks)
        deleted = initial_count != final_count
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - delete_feedback: {deleted}")
        return deleted

    async def upsert_feedback(self, feedback: Feedback) -> str:
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - upsert_feedback called with feedback: {feedback}")
        # If feedback has an ID, update existing, else create new
        if hasattr(feedback, 'id') and feedback.id:
            for i, existing_feedback in enumerate(self.feedbacks):
                if getattr(existing_feedback, 'id', None) == feedback.id:
                    self.feedbacks[i] = feedback
                    if self.verbosity:
                        logger.info(f"InMemoryDataLayer - Feedback updated: {feedback.id}")
                    return feedback.id
        # Create new feedback
        if not hasattr(feedback, 'id') or not feedback.id:
            feedback.id = str(uuid.uuid4())
        self.feedbacks.append(feedback)
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - Feedback created: {feedback.id}")
        return feedback.id

    async def create_element(self, element: "Element"):
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - create_element called with element: {element}")
        # Convert Element to ElementDict
        element_dict = {
            "id": getattr(element, 'id', str(uuid.uuid4())),
            "type": getattr(element, 'type', ''),
            "name": getattr(element, 'name', ''),
            "display": getattr(element, 'display', 'side'),
            "url": getattr(element, 'url', ''),
            "objectKey": getattr(element, 'objectKey', None),
            "size": getattr(element, 'size', None),
            "page": getattr(element, 'page', None),
            "language": getattr(element, 'language', None),
            "forId": getattr(element, 'forId', None),
            "mime": getattr(element, 'mime', ''),
        }
        self.elements.append(element_dict)
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - Element created: {element_dict['id']}")

    async def get_element(self, thread_id: str, element_id: str) -> Optional["ElementDict"]:
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - get_element called with thread_id: {thread_id}, element_id: {element_id}")
        element = next((e for e in self.elements if e.get("id") == element_id), None)
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - get_element returning: {element}")
        return element

    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - delete_element called with element_id: {element_id}, thread_id: {thread_id}")
        initial_count = len(self.elements)
        self.elements = [e for e in self.elements if e.get("id") != element_id]
        final_count = len(self.elements)
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - Element deletion: {initial_count} -> {final_count} elements")

    async def create_step(self, step_dict: "StepDict"):
        # Follow SQLAlchemy pattern: update_thread ensures thread exists via upsert
        await self.update_thread(step_dict["threadId"])
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - create_step called with step_dict: {step_dict}")
        # Ensure step has required fields
        if "id" not in step_dict:
            step_dict["id"] = str(uuid.uuid4())
        if "createdAt" not in step_dict:
            step_dict["createdAt"] = datetime.now().isoformat()
        self.steps.append(step_dict)
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - Step created: {step_dict['id']}. Total steps: {len(self.steps)}")

    async def update_step(self, step_dict: "StepDict"):
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - update_step called with step_dict: {step_dict}")
        step_id = step_dict.get("id")
        if not step_id:
            if self.verbosity:
                logger.error("InMemoryDataLayer - update_step: step_dict missing id")
            return
            
        # Check if step exists
        step_exists = False
        for i, step in enumerate(self.steps):
            if step.get("id") == step_id:
                # Merge the existing step with updates
                updated_step = {**step, **step_dict}
                self.steps[i] = updated_step
                if self.verbosity:
                    logger.info(f"InMemoryDataLayer - Step {step_id} updated")
                step_exists = True
                break
        
        # AUTO-CREATE STEP IF IT DOESN'T EXIST
        if not step_exists:
            if self.verbosity:
                logger.warning(f"InMemoryDataLayer - Step {step_id} not found, creating it from update")
            # Ensure required fields are present
            if "createdAt" not in step_dict:
                step_dict["createdAt"] = datetime.now().isoformat()
            if "start" not in step_dict:
                step_dict["start"] = step_dict["createdAt"]
            self.steps.append(step_dict)
            if self.verbosity:
                logger.info(f"Step created from update: {step_id}")

    async def delete_step(self, step_id: str):
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - delete_step called with step_id: {step_id}")
        initial_count = len(self.steps)
        self.steps = [s for s in self.steps if s.get("id") != step_id]
        final_count = len(self.steps)
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - Step deletion: {initial_count} -> {final_count} steps")

    async def get_thread_author(self, thread_id: str) -> str:
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - get_thread_author called with thread_id: {thread_id}")
        thread = await self.get_thread(thread_id)
        if thread:
            user_identifier = thread.get("userIdentifier")
            if user_identifier:
                if self.verbosity:
                    logger.info(f"InMemoryDataLayer - Returning author: {user_identifier}")
                return user_identifier
        if self.verbosity:
            logger.warning(f"InMemoryDataLayer - Author not found for thread {thread_id}")
        raise ValueError(f"Author not found for thread_id {thread_id}")

    async def create_thread(self, thread: "ThreadDict") -> str:
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - create_thread called with thread: {thread}")
        # Ensure thread has required fields
        if "id" not in thread:
            thread["id"] = str(uuid.uuid4())
        if "createdAt" not in thread:
            thread["createdAt"] = datetime.now().isoformat()
        
        self.threads.append(thread)
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - Thread created with id: {thread['id']}. Total threads: {len(self.threads)}")
        return thread["id"]

    async def delete_thread(self, thread_id: str):
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - delete_thread called with thread_id: {thread_id}")
        initial_thread_count = len(self.threads)
        initial_step_count = len(self.steps)
        initial_element_count = len(self.elements)
        
        # Delete thread
        self.threads = [t for t in self.threads if t.get("id") != thread_id]
        # Delete steps associated with thread
        self.steps = [s for s in self.steps if s.get("threadId") != thread_id]
        # Delete elements associated with thread
        self.elements = [e for e in self.elements if e.get("threadId") != thread_id]
        
        final_thread_count = len(self.threads)
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - Thread deletion: {initial_thread_count} -> {final_thread_count} threads, "
                       f"removed {initial_step_count - len(self.steps)} steps, "
                       f"removed {initial_element_count - len(self.elements)} elements")

    async def list_threads(
        self, pagination: "Pagination", filters: "ThreadFilter"
    ) -> "PaginatedResponse[ThreadDict]":
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - list_threads - pagination: first={pagination.first}, cursor={pagination.cursor}")
            logger.info(f"InMemoryDataLayer - list_threads - filters: userId={filters.userId}, search={filters.search}")
            logger.info(f"InMemoryDataLayer - list_threads - Total threads available: {len(self.threads)}")
        
        # Start with all threads
        filtered_threads = self.threads.copy()
        
        # Apply filters
        if filters:
            # Filter by user ID
            if filters.userId:
                filtered_threads = [t for t in filtered_threads if t.get("userId") == filters.userId]
            
            # Filter by search term in thread name
            if filters.search:
                search_lower = filters.search.lower()
                filtered_threads = [t for t in filtered_threads 
                                  if t.get("name") and search_lower in t["name"].lower()]
            
            # Filter by feedback (you'll need to implement this based on your feedback structure)
            if filters.feedback is not None:
                # This would require linking feedback to threads
                if self.verbosity:
                    logger.info(f"InMemoryDataLayer - Feedback filtering not yet implemented: {filters.feedback}")
        
        # Sort threads by createdAt descending (newest first)
        filtered_threads.sort(key=lambda x: x.get("createdAt", ""), reverse=True)
        
        # Apply pagination
        start_index = 0
        if pagination.cursor:
            # Find the index of the cursor thread
            for i, thread in enumerate(filtered_threads):
                if thread.get("id") == pagination.cursor:
                    start_index = i + 1
                    break
        
        end_index = start_index + pagination.first
        paginated_threads = filtered_threads[start_index:end_index]
        
        # Add steps and elements to each thread
        for thread in paginated_threads:
            thread_id = thread["id"]
            thread["steps"] = [s for s in self.steps if s.get("threadId") == thread_id]
            thread["elements"] = [e for e in self.elements if e.get("threadId") == thread_id]
            # Sort steps by createdAt to match other implementations
            thread["steps"].sort(key=lambda x: x.get("createdAt") or "")
        
        # Create page info
        has_next_page = len(filtered_threads) > end_index
        start_cursor = paginated_threads[0]["id"] if paginated_threads else None
        end_cursor = paginated_threads[-1]["id"] if paginated_threads else None

        page_info = PageInfo(
            hasNextPage=has_next_page,
            startCursor=start_cursor,
            endCursor=end_cursor
        )

        if self.verbosity:
            logger.info(f"InMemoryDataLayer - list_threads - Returning {len(paginated_threads)} threads, hasNextPage: {page_info.hasNextPage}")
            logger.info(f"InMemoryDataLayer - list_threads - Start cursor: {page_info.startCursor}, End cursor: {page_info.endCursor}")
        return PaginatedResponse(pageInfo=page_info, data=paginated_threads)

    async def get_thread(self, thread_id: str) -> "Optional[ThreadDict]":
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - get_thread called with thread_id: {thread_id}")
        thread = next((t for t in self.threads if t.get("id") == thread_id), None)
        if thread:
            # Add steps and elements to the thread
            thread["steps"] = [s for s in self.steps if s.get("threadId") == thread_id]
            thread["elements"] = [e for e in self.elements if e.get("threadId") == thread_id]
            # Sort steps by createdAt to match other implementations
            thread["steps"].sort(key=lambda x: x.get("createdAt") or "")
            if self.verbosity:
                logger.info(f"InMemoryDataLayer - get_thread found thread with {len(thread['steps'])} steps and {len(thread['elements'])} elements")
        else:
            if self.verbosity:
                logger.info(f"InMemoryDataLayer - get_thread: thread {thread_id} not found")
        return thread

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        if self.verbosity:
            logger.info(f"InMemoryDataLayer - update_thread called with thread_id: {thread_id}, name: {name}, user_id: {user_id}, metadata: {metadata}, tags: {tags}")
        
        # Find existing thread or create new one
        thread_index = None
        for i, thread in enumerate(self.threads):
            if thread.get("id") == thread_id:
                thread_index = i
                break
        
        if thread_index is not None:
            # Update existing thread
            thread = self.threads[thread_index]
            if name is not None:
                thread["name"] = name
            if user_id is not None:
                thread["userId"] = user_id
            if metadata is not None:
                thread["metadata"] = metadata
            if tags is not None:
                thread["tags"] = tags
            self.threads[thread_index] = thread
            if self.verbosity:
                logger.info(f"InMemoryDataLayer - Thread {thread_id} updated")
        else:
            # Create new thread - get user from context if not provided
            if user_id is None and hasattr(context, 'session') and context.session and context.session.user:
                user_id = context.session.user.id
                user_identifier = context.session.user.identifier
            else:
                user_identifier = None

            thread = ThreadDict(
                id=thread_id,
                createdAt=datetime.now().isoformat(),
                name=name,
                userId=user_id,
                userIdentifier=user_identifier,
                tags=tags,
                metadata=metadata or {},
                steps=[],
                elements=[]
            )
            self.threads.append(thread)
            if self.verbosity:
                logger.info(f"InMemoryDataLayer - Thread {thread_id} created with userId={user_id}, userIdentifier={user_identifier}")

    async def close(self):
        if self.verbosity:
            logger.info("InMemoryDataLayer - close called")
            
    async def build_debug_url(self) -> str:
        if self.verbosity:
            logger.info("build_debug_url called")
        return "simple-data-layer-debug-info"